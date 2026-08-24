"""Build a relocatable, packed Python environment to ship to remote workers.

Most commonly this is the poncho package handed to
`TaskVineDistributor(environment=...)` (see taskvine_distributor.py and
DOC.md's "Packaging the environment for TaskVine workers"), but nothing
here is TaskVine-specific: `get_environment()` just resolves a package spec
to a tarball path on disk. Any current or future `Distributor` that wants a
shipped environment can use it the same way; a distributor whose workers
already share vine_reduce's filesystem (e.g. LocalDistributor) simply has
no use for the result.

Adapted from TopEFT/topcoffea's `topcoffea/modules/remote_environment.py`
(https://github.com/TopEFT/topcoffea), generalized so the default package
set is vine_reduce itself rather than a fixed HEP analysis stack - callers
add whatever else their workers need via extra_conda/extra_pip.

Building goes through `poncho_package_create` (part of `ndcctools`/cctools,
same as TaskVine itself - see the `conda` extra in pyproject.toml), which
takes a conda+pip spec and produces a single relocatable tarball. Two things
make repeated calls cheap:
  - Results are cached on disk, keyed by a hash of the resolved spec plus
    the state of any locally-editable packages being watched (see below).
    A cache hit just returns the existing tarball path immediately.
  - Any locally-editable pip install found via `pip list --editable`
    (vine_reduce itself by default, or whatever else is listed in
    pip_local_to_watch) has its git commit hash folded into that cache key,
    and unstaged changes to the paths being watched are treated as "always
    rebuild" (or raise UnstagedChanges, depending on `unstaged=`) - so a
    tarball never silently ships stale code from an editable checkout.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PY_VERSION = "{}.{}.{}".format(*sys.version_info[:3])

_DEFAULT_CACHE_DIR = Path.cwd() / "vine_reduce-envs"

_DEFAULT_MODULES: dict[str, Any] = {
    "conda": {
        "channels": ["conda-forge"],
        "packages": [f"python={_PY_VERSION}", "pip", "conda-pack", "ndcctools>=7.17.1"],
    },
    # Not yet on PyPI, so installed straight from its own repository (a
    # plain PEP 508 direct reference, same as this project's own
    # [tool.pixi.pypi-dependencies] entry for itself) - switch to a bare
    # "vine_reduce" once it's published there.
    "pip": ["vine_reduce @ git+https://github.com/cooperative-computing-lab/vine_reduce.git"],
}

# package name (as reported by `pip list --editable`, i.e. its distribution
# name, not an install spec) -> paths relative to its repo root whose git
# status decides whether an editable install of that package counts as
# "changed" for the cache key - see _local_pip_commits.
_DEFAULT_PIP_LOCAL_TO_WATCH: dict[str, list[str]] = {"vine_reduce": ["src", "pyproject.toml"]}


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


def _pin_versions_from_current_env(spec: dict[str, Any]) -> dict[str, Any]:
    """For any package in `spec` that isn't already version-pinned, pin it to
    whatever version is active in the current conda environment, if any -
    so the packed environment matches local development rather than
    resolving possibly-different versions at pack time."""
    with tempfile.NamedTemporaryFile() as f:
        subprocess.check_call(
            ["conda", "env", "export", "--json"], stdout=f, stdin=subprocess.DEVNULL
        )
        with open(f.name) as spec_file:
            current_spec = json.load(spec_file)
        current_spec["pinning"] = {"conda": _current_conda_package_versions()}

        dependencies = current_spec.get("dependencies", [])
        conda_deps = {
            re.sub("[!~=<>].*$", "", x): x for x in dependencies if not isinstance(x, dict)
        }
        pip_deps = {
            re.sub("[!~=<>].*$", "", y): y
            for x in dependencies
            if isinstance(x, dict) and "pip" in x
            for y in x["pip"]
        }

        for i, package in enumerate(spec["conda"]["packages"]):
            if not re.search("[!~=<>].*$", package) and package in conda_deps:
                spec["conda"]["packages"][i] = conda_deps[package]

        for i, package in enumerate(spec["pip"]):
            if not re.search("[!~=<>].*$", package) and package in pip_deps:
                spec["pip"][i] = pip_deps[package]

    return spec


def _create_env(env_path: str, spec: dict[str, Any], force: bool = False) -> str:
    if force:
        logger.info("Forcing rebuild of %s", env_path)
        Path(env_path).unlink(missing_ok=True)
    elif Path(env_path).exists():
        logger.info("Found in cache: %s", env_path)
        return env_path

    logger.info("Checking current conda environment")
    spec = _pin_versions_from_current_env(spec)

    with tempfile.NamedTemporaryFile() as f:
        packages_json = json.dumps(spec)
        logger.info("base env specification: %s", packages_json)
        f.write(packages_json.encode())
        f.flush()
        logger.info("Creating environment %s", env_path)

        try:
            subprocess.check_output(
                ["poncho_package_create", f.name, env_path], stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            logger.error("poncho package creation failed with code %s", e.returncode)
            logger.error(e.output.decode())
            raise

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
    paths_by_package: dict[str, str], pip_local_to_watch: dict[str, list[str]]
) -> dict[str, str]:
    """For each editable package, the git commit of its checkout, or the
    sentinel "HEAD" if the watched paths have uncommitted changes (or the
    checkout isn't a git repo at all - safest default is to always rebuild)."""
    commits: dict[str, str] = {}
    for package, path in paths_by_package.items():
        try:
            watch_paths = pip_local_to_watch.get(package)
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


def _combined_commit_key(paths_by_package: dict[str, str], commits: dict[str, str]) -> str:
    if not commits:
        return "fixed"
    values = [commits[p] for p in paths_by_package]
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
    extra_conda: list[str] | None = None,
    extra_pip: list[str] | None = None,
    pip_local_to_watch: dict[str, list[str]] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    unstaged: str = "rebuild",
    cache_size: int = 3,
) -> str:
    """Resolve a packed environment tarball, building (or rebuilding) it
    with poncho_package_create if needed, and return its path - suitable to
    pass straight to `TaskVineDistributor(environment=...)`.

    extra_conda/extra_pip add packages beyond the vine_reduce-only default -
    each pip entry can be a plain requirement or any pip-recognized install
    spec (a version pin, a local path, a `name @ git+URL` direct reference,
    ...), e.g. `extra_pip=["/path/to/my-analysis-repo"]` to also pack a
    locally-checked-out analysis package that isn't published anywhere.

    pip_local_to_watch merges on top of the default (vine_reduce's own
    `src`/pyproject.toml): package name (as reported by
    `pip list --editable`, not an install spec - see _find_editable_pip_installs)
    -> list of paths, relative to that package's repo root, to `git status`.
    If any of those packages is currently installed editable *and* has
    uncommitted changes there, the build is treated as stale (see
    `unstaged` below) regardless of whether it was also named in extra_pip.

    unstaged controls what happens when a watched editable install has
    uncommitted changes: "rebuild" (default) forces a fresh build, "fail"
    raises UnstagedChanges instead.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec: dict[str, Any] = {
        "conda": {
            "channels": list(_DEFAULT_MODULES["conda"]["channels"]),
            "packages": list(_DEFAULT_MODULES["conda"]["packages"]),
        },
        "pip": list(_DEFAULT_MODULES["pip"]),
    }
    watch = dict(_DEFAULT_PIP_LOCAL_TO_WATCH)
    if pip_local_to_watch:
        watch.update(pip_local_to_watch)

    if extra_conda:
        spec["conda"]["packages"].extend(extra_conda)
    if extra_pip:
        spec["pip"].extend(extra_pip)

    packages_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]

    paths_by_package = _find_editable_pip_installs()
    commits = _local_pip_commits(paths_by_package, watch)
    pip_check = _combined_commit_key(paths_by_package, commits)

    env_path = str(cache_dir / f"env_spec_{packages_hash}_edit_{pip_check}.tar.gz")
    _trim_cache(cache_dir, cache_size, env_path)

    if pip_check == "HEAD":
        changed = [p for p, c in commits.items() if c == "HEAD"]
        if unstaged == "fail":
            raise UnstagedChanges(changed)
        if unstaged == "rebuild":
            force = True
            logger.warning(
                "Rebuilding environment because of unstaged changes in: %s",
                ", ".join(Path(paths_by_package[p]).name for p in changed),
            )

    return _create_env(env_path, spec, force)
