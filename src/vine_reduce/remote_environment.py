"""Build a relocatable, packed conda environment tarball to ship to remote workers, e.g. via
`TaskVineDistributor(environment=...)`. Not TaskVine-specific: get_environment() just returns a
tarball path.

Adapted from TopEFT/topcoffea's topcoffea/modules/remote_environment.py
(https://github.com/TopEFT/topcoffea). Packs whatever is installed in the calling conda
environment ($CONDA_PREFIX by default) with `poncho_package_create` (ndcctools; the `conda`
extra).

Tarballs are cached on disk, keyed by a hash of the environment's installed packages plus the
git state of locally-editable installs (commit hash; uncommitted changes force a rebuild or raise
UnstagedChanges), so a tarball never ships stale editable code. Editable installs can't be packed
as-is, so they are temporarily reinstalled non-editable for packing, then restored.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.cwd() / "vine_reduce-envs"

# package name (as in `pip list --editable`) -> paths, relative to its repo root, whose git status
# marks an editable install as changed for the cache key.
_DEFAULT_PIP_EDITABLE: dict[str, list[str]] = {"vine_reduce": ["src", "pyproject.toml"]}

# Cache-key sentinel meaning "always rebuild": uncommitted changes, or a commit that can't be
# determined. Not a real commit hash, so it can't collide with one.
_DIRTY = "dirty"


class UnstagedChanges(Exception):
    """Raised by get_environment(unstaged="fail") when a watched, locally-
    editable package has uncommitted changes."""


def _current_conda_package_versions(conda_env_path: str | None = None) -> dict[str, str]:
    """name -> "name=version=build" for every package in the environment: conda packages from
    its conda-meta records, pip packages from its site-packages. Read directly rather than via
    `conda list`, so it needs no `conda` executable (pixi envs have none, and `pixi run` drops
    conda from PATH)."""
    prefix = Path(conda_env_path or os.environ.get("CONDA_PREFIX") or sys.prefix)

    versions = {}
    for record in (prefix / "conda-meta").glob("*.json"):
        pkg = json.loads(record.read_text())
        versions[pkg["name"]] = f"{pkg['name']}={pkg['version']}={pkg['build']}"

    site_packages = glob.glob(str(prefix / "lib" / "python*" / "site-packages"))
    for dist in importlib.metadata.distributions(path=site_packages):
        name = f"pypi:{dist.metadata['Name']}"
        versions[name] = f"{name}={dist.version}"

    return versions


def _environment_state_hash(conda_env_path: str) -> str:
    """Hash of every package installed in conda_env_path (conda and pip), for the cache key."""
    versions = _current_conda_package_versions(conda_env_path)
    return hashlib.sha256("".join(sorted(versions.values())).encode()).hexdigest()[:8]


def _check_pack_dependencies() -> None:
    """Raise a clear error if poncho_package_create or conda-pack is missing."""
    missing = []
    if shutil.which("poncho_package_create") is None:
        missing.append("ndcctools (provides poncho_package_create)")
    if importlib.util.find_spec("conda_pack") is None:
        missing.append("conda-pack")
    if missing:
        raise RuntimeError(
            "Cannot build a packed environment, missing: "
            + ", ".join(missing)
            + ". Install them (e.g. `pixi add ndcctools conda-pack`, or this project's "
            "'conda' extra) before calling get_environment()."
        )


def _create_env(
    env_path: str, conda_env_path: str, paths_by_package: dict[str, str], force: bool = False
) -> str:
    if force:
        logger.info("Forcing rebuild of %s", env_path)
        Path(env_path).unlink(missing_ok=True)
    elif Path(env_path).exists():
        logger.info("Found in cache: %s", env_path)
        return env_path

    _check_pack_dependencies()

    for package, path in paths_by_package.items():
        logger.info("Reinstalling %s non-editable for packing", package)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", path],
            stdin=subprocess.DEVNULL,
        )

    try:
        logger.info("Creating environment %s from %s", env_path, conda_env_path)
        subprocess.check_output(
            ["poncho_package_create", conda_env_path, env_path], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        logger.error("poncho package creation failed with code %s", e.returncode)
        logger.error(e.output.decode())
        raise
    finally:
        for package, path in paths_by_package.items():
            logger.info("Reinstalling %s editable", package)
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    "-e",
                    path,
                ],
                stdin=subprocess.DEVNULL,
            )

    return env_path


def _find_editable_pip_installs() -> dict[str, str]:
    """package name -> checkout path for every editable pip install."""
    paths_by_package = {}
    for dist in importlib.metadata.distributions():
        raw_direct_url = dist.read_text("direct_url.json")
        if raw_direct_url is None:
            continue
        direct_url = json.loads(raw_direct_url)
        if not direct_url.get("dir_info", {}).get("editable", False):
            continue
        location = url2pathname(urlparse(direct_url["url"]).path)
        paths_by_package[dist.metadata["Name"]] = location
    return paths_by_package


def _git_head(path: str) -> str:
    """The current commit hash of the git checkout at path."""
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, stdin=subprocess.DEVNULL)
        .decode()
        .rstrip()
    )


def _git_dirty(path: str, pathspecs: list[str]) -> str:
    """`git status --porcelain` for path restricted to pathspecs; empty means clean. Retries
    unrestricted if git rejects the pathspecs."""
    status_cmd = ["git", "status", "--porcelain", "--untracked-files=no"]
    try:
        return (
            subprocess.check_output(status_cmd + pathspecs, cwd=path, stdin=subprocess.DEVNULL)
            .decode()
            .rstrip()
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "Could not apply git paths-to-watch filters for %s; trying without them", path
        )
        return (
            subprocess.check_output(status_cmd, cwd=path, stdin=subprocess.DEVNULL)
            .decode()
            .rstrip()
        )


def _local_pip_commits(
    paths_by_package: dict[str, str], pip_editable: dict[str, list[str]]
) -> dict[str, str]:
    """Per editable package: its git commit, or _DIRTY if the watched paths have uncommitted
    changes or the commit can't be determined (e.g. not a git repo)."""
    commits: dict[str, str] = {}
    for package, path in paths_by_package.items():
        try:
            watch_paths = pip_editable.get(package)
            pathspecs = [f":(top){p}" for p in watch_paths] if watch_paths else []

            commit = _git_head(path)
            changed = _git_dirty(path, pathspecs)

            if changed:
                logger.warning("Found unstaged changes in %s:\n%s", path, changed)
                commits[package] = _DIRTY
            else:
                commits[package] = commit
        except Exception as e:
            logger.warning("Could not get current commit of %r: %s", path, e)
            commits[package] = _DIRTY
    return commits


def _combined_commit_key(commits: dict[str, str]) -> str:
    """Editable-installs half of the cache key: a hash of all commits, "fixed" if there are none,
    or _DIRTY if any has uncommitted changes."""
    if not commits:
        return "fixed"
    values = list(commits.values())
    if _DIRTY in values:
        # always rebuild rather than trust a cache entry that might be stale
        return _DIRTY
    return hashlib.sha256("".join(values).encode()).hexdigest()[:8]


def _trim_cache(cache_dir: Path, cache_size: int, *keep: str) -> None:
    envs = sorted(
        glob.glob(os.path.join(cache_dir, "env_*.tar.gz")), key=lambda f: -os.stat(f).st_mtime
    )
    for f in envs[cache_size:]:
        if f not in keep:
            logger.info("Trimming cached environment file %s", f)
            os.remove(f)


def get_environment(
    conda_env_path: str | Path | None = None,
    pip_editable: dict[str, list[str]] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    unstaged: str = "rebuild",
    cache_size: int = 3,
) -> str:
    """Return the path of a packed environment tarball, building it with poncho_package_create
    if not cached. Suitable for `TaskVineDistributor(environment=...)`.

    conda_env_path defaults to $CONDA_PREFIX; exactly what is installed there is packed, so
    install any packages workers need beforehand.

    pip_editable merges over the default (vine_reduce's `src`/pyproject.toml): package name (as in
    `pip list --editable`) -> paths, relative to its repo root, whose `git status` decides
    whether that package counts as changed for the cache key. Every editable install is repacked
    non-editable and restored afterwards, whether or not it is named here; an unnamed one is
    watched over its whole repo.

    unstaged: "rebuild" (default) forces a fresh build when a watched editable install has
    uncommitted changes; "fail" raises UnstagedChanges instead.
    """
    if unstaged not in ("rebuild", "fail"):
        raise ValueError(f"unstaged must be 'rebuild' or 'fail', not {unstaged!r}")

    conda_env_path = str(conda_env_path) if conda_env_path else os.environ.get("CONDA_PREFIX")
    if not conda_env_path:
        raise RuntimeError(
            "No conda environment to pack: pass conda_env_path or activate one (so "
            "$CONDA_PREFIX is set)"
        )

    cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    watch = dict(_DEFAULT_PIP_EDITABLE)
    if pip_editable:
        watch.update(pip_editable)

    paths_by_package = _find_editable_pip_installs()
    commits = _local_pip_commits(paths_by_package, watch)
    pip_check = _combined_commit_key(commits)

    env_hash = _environment_state_hash(conda_env_path)
    env_path = str(cache_dir / f"env_{env_hash}_edit_{pip_check}.tar.gz")
    _trim_cache(cache_dir, cache_size, env_path)

    if pip_check == _DIRTY:
        changed = [p for p, c in commits.items() if c == _DIRTY]
        if unstaged == "fail":
            raise UnstagedChanges(changed)
        force = True
        logger.warning(
            "Rebuilding environment because of unstaged changes in: %s",
            ", ".join(Path(paths_by_package[p]).name for p in changed),
        )

    return _create_env(env_path, conda_env_path, paths_by_package, force)
