"""Build a relocatable, packed Python environment to ship to remote workers.

Most commonly this is the poncho package handed to
`TaskVineDistributor(environment=...)` (see taskvine_distributor.py and the
README's "Packaging an environment for remote workers"), but nothing here
is TaskVine-specific: `get_environment()` just resolves a conda environment
to a tarball path on disk. Any current or future `Distributor` that wants a
shipped environment can use it the same way; a distributor whose workers
already share vine_reduce's filesystem (e.g. LocalDistributor) simply has
no use for the result.

Adapted from TopEFT/topcoffea's `topcoffea/modules/remote_environment.py`
(https://github.com/TopEFT/topcoffea), simplified to pack whatever is
already installed in the calling conda environment (normally $CONDA_PREFIX)
rather than resolving a separate package spec - add whatever your workers
need by installing it into that environment before calling get_environment,
not through this module.

Building goes through `poncho_package_create` (part of `ndcctools`/cctools,
same as TaskVine itself - see the `conda` extra in pyproject.toml), which
packs a conda environment directory into a single relocatable tarball. Two
things make repeated calls cheap:
  - Results are cached on disk, keyed by a hash of the environment's
    installed packages plus the state of any locally-editable packages (see
    below). A cache hit just returns the existing tarball path immediately.
  - Any locally-editable pip install found via `pip list --editable`
    (vine_reduce itself by default, or whatever else is listed in
    pip_editable) has its git commit hash folded into that cache key, and
    unstaged changes to the paths being watched are treated as "always
    rebuild" (or raise UnstagedChanges, depending on `unstaged=`) - so a
    tarball never silently ships stale code from an editable checkout.

Editable installs can't be packed as-is - conda-pack needs real files in
site-packages, not a .pth pointing back at a checkout that won't exist on a
remote worker - so any package currently installed editable is temporarily
reinstalled non-editable for the pack step, then reinstalled editable again
immediately afterwards (see _create_env).
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.cwd() / "vine_reduce-envs"

# package name (as reported by `pip list --editable`, i.e. its distribution
# name, not an install spec) -> paths relative to its repo root whose git
# status decides whether an editable install of that package counts as
# "changed" for the cache key - see _local_pip_commits.
_DEFAULT_PIP_EDITABLE: dict[str, list[str]] = {"vine_reduce": ["src", "pyproject.toml"]}


class UnstagedChanges(Exception):
    """Raised by get_environment(unstaged="fail") when a watched, locally-
    editable package has uncommitted changes."""


def _current_conda_package_versions(conda_env_path: str | None = None) -> dict[str, str]:
    conda_env_path = conda_env_path or os.environ.get("CONDA_PREFIX")

    cmd = ["conda", "list", "--export", "--json"]
    if conda_env_path:
        cmd += ["--prefix", conda_env_path]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    raw_pkgs = json.loads(proc.stdout.decode())

    return {
        pkg["name"]: f"{pkg['name']}={pkg['version']}={pkg['build_string']}" for pkg in raw_pkgs
    }


def _environment_state_hash(conda_env_path: str) -> str:
    """Hash of every package currently installed in conda_env_path (conda
    and pip packages alike - pip installs show up in `conda list` too), used
    as part of the cache key so a rebuild is triggered whenever the
    environment's contents change."""
    versions = _current_conda_package_versions(conda_env_path)
    return hashlib.sha256("".join(sorted(versions.values())).encode()).hexdigest()[:8]


def _check_pack_dependencies() -> None:
    """Raise a clear error if poncho_package_create (from ndcctools) or its
    conda-pack dependency aren't installed, rather than failing deep inside
    a subprocess call with a cryptic error."""
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
    env_path: str, conda_env_path: str, editable: dict[str, str], force: bool = False
) -> str:
    if force:
        logger.info("Forcing rebuild of %s", env_path)
        Path(env_path).unlink(missing_ok=True)
    elif Path(env_path).exists():
        logger.info("Found in cache: %s", env_path)
        return env_path

    _check_pack_dependencies()

    for package, path in editable.items():
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
        for package, path in editable.items():
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
    """package name -> local checkout path, for every package currently
    `pip install -e`d in this Python's environment."""
    raw = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--editable"], stdin=subprocess.DEVNULL
    ).decode()

    # first two lines are a header ("Package Version Editable project location", "----")
    paths_by_package = {}
    for line in raw.splitlines()[2:]:
        if not line:
            continue
        package, _version, location = line.split()
        paths_by_package[package] = location
    return paths_by_package


def _local_pip_commits(
    paths_by_package: dict[str, str], pip_editable: dict[str, list[str]]
) -> dict[str, str]:
    """For each editable package, the git commit of its checkout, or the
    sentinel "HEAD" if the watched paths have uncommitted changes (or the
    checkout isn't a git repo at all - safest default is to always rebuild)."""
    commits: dict[str, str] = {}
    for package, path in paths_by_package.items():
        try:
            watch_paths = pip_editable.get(package)
            pathspecs = [f":(top){p}" for p in watch_paths] if watch_paths else []

            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=path, stdin=subprocess.DEVNULL
                )
                .decode()
                .rstrip()
            )

            status_cmd = ["git", "status", "--porcelain", "--untracked-files=no"]
            try:
                changed = (
                    subprocess.check_output(
                        status_cmd + pathspecs, cwd=path, stdin=subprocess.DEVNULL
                    )
                    .decode()
                    .rstrip()
                )
            except subprocess.CalledProcessError:
                logger.warning(
                    "Could not apply git paths-to-watch filters for %s; trying without them", path
                )
                changed = (
                    subprocess.check_output(status_cmd, cwd=path, stdin=subprocess.DEVNULL)
                    .decode()
                    .rstrip()
                )

            if changed:
                logger.warning("Found unstaged changes in %s:\n%s", path, changed)
                commits[package] = "HEAD"
            else:
                commits[package] = commit
        except Exception as e:
            logger.warning("Could not get current commit of %r: %s", path, e)
            commits[package] = "HEAD"
    return commits


def _combined_commit_key(commits: dict[str, str]) -> str:
    """The editable-installs half of the cache key: one hash over every
    editable package's commit, or the "HEAD" sentinel if any of them has
    uncommitted changes (see _local_pip_commits)."""
    if not commits:
        return "fixed"
    values = list(commits.values())
    if "HEAD" in values:
        # always rebuild rather than trust a cache entry that might be stale
        return "HEAD"
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
    """Resolve a packed environment tarball, building (or rebuilding) it
    with poncho_package_create if needed, and return its path - suitable to
    pass straight to `TaskVineDistributor(environment=...)`.

    conda_env_path defaults to $CONDA_PREFIX: the environment packed is
    whatever is currently installed there, nothing more. Add packages your
    workers need by installing them into that environment (conda install,
    pip install, a pixi dependency, ...) before calling this.

    pip_editable merges on top of the default (vine_reduce's own
    `src`/pyproject.toml): package name (as reported by
    `pip list --editable`, not an install spec - see _find_editable_pip_installs)
    -> list of paths, relative to that package's repo root, to `git status`.
    Every package currently installed editable is reinstalled non-editable
    for the pack step and back to editable afterwards, regardless of whether
    it's named here; pip_editable only narrows which paths are watched to
    decide if that package counts as "changed" for the cache key (a package
    installed editable but not named here still gets watched, over its
    whole repo).

    unstaged controls what happens when a watched editable install has
    uncommitted changes: "rebuild" (default) forces a fresh build, "fail"
    raises UnstagedChanges instead.
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

    if pip_check == "HEAD":
        changed = [p for p, c in commits.items() if c == "HEAD"]
        if unstaged == "fail":
            raise UnstagedChanges(changed)
        force = True
        logger.warning(
            "Rebuilding environment because of unstaged changes in: %s",
            ", ".join(Path(paths_by_package[p]).name for p in changed),
        )

    return _create_env(env_path, conda_env_path, paths_by_package, force)
