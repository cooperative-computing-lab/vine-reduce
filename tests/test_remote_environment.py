from __future__ import annotations

import os
import subprocess
import time

import pytest

from vine_reduce import remote_environment as re_mod
from vine_reduce.remote_environment import (
    UnstagedChanges,
    _check_pack_dependencies,
    _combined_commit_key,
    _local_pip_commits,
    _trim_cache,
    get_environment,
)


def test_combined_commit_key_with_no_editable_packages_is_fixed():
    assert _combined_commit_key({}) == "fixed"


def test_combined_commit_key_is_head_if_any_package_is_head():
    assert _combined_commit_key({"a": "abc123", "b": "HEAD"}) == "HEAD"


def test_combined_commit_key_is_deterministic_hash_of_clean_commits():
    commits = {"a": "abc123", "b": "def456"}
    first = _combined_commit_key(commits)
    second = _combined_commit_key(commits)
    assert first == second
    assert first != "HEAD" and first != "fixed"


def _git(path, *args):
    subprocess.check_call(
        ["git", *args], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "pkg.py").write_text("x = 1\n")
    (repo / "pyproject.toml").write_text("[project]\nname='pkg'\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_local_pip_commits_reports_commit_for_clean_checkout(git_repo):
    commits = _local_pip_commits({"pkg": str(git_repo)}, {"pkg": ["src", "pyproject.toml"]})
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo).decode().rstrip()
    assert commits["pkg"] == expected


def test_local_pip_commits_reports_head_for_unstaged_watched_change(git_repo):
    (git_repo / "src" / "pkg.py").write_text("x = 2\n")
    commits = _local_pip_commits({"pkg": str(git_repo)}, {"pkg": ["src", "pyproject.toml"]})
    assert commits["pkg"] == "HEAD"


def test_local_pip_commits_ignores_unwatched_paths(git_repo):
    (git_repo / "README.md").write_text("not watched\n")
    commits = _local_pip_commits({"pkg": str(git_repo)}, {"pkg": ["src", "pyproject.toml"]})
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo).decode().rstrip()
    assert commits["pkg"] == expected


def test_trim_cache_keeps_only_newest_n(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"env_{i}.tar.gz"
        p.write_text("x")
        os.utime(p, (time.time() + i, time.time() + i))
        paths.append(str(p))

    _trim_cache(tmp_path, 2)

    remaining = {os.path.basename(p) for p in tmp_path.glob("env_*.tar.gz")}
    assert remaining == {"env_4.tar.gz", "env_3.tar.gz"}


def test_trim_cache_never_removes_the_kept_paths(tmp_path):
    p = tmp_path / "env_old.tar.gz"
    p.write_text("x")
    os.utime(p, (0, 0))

    _trim_cache(tmp_path, 0, str(p))

    assert p.exists()


def test_get_environment_reuses_cache_without_rebuilding(tmp_path, monkeypatch):
    build_calls = []

    def fake_create(env_path, conda_env_path, editable, force=False):
        build_calls.append(env_path)
        with open(env_path, "wb"):
            pass
        return env_path

    monkeypatch.setattr(re_mod, "_find_editable_pip_installs", lambda: {})
    monkeypatch.setattr(re_mod, "_environment_state_hash", lambda conda_env_path: "abc123")
    monkeypatch.setattr(re_mod, "_create_env", fake_create)

    first = get_environment(conda_env_path="/fake/env", cache_dir=tmp_path)
    second = get_environment(conda_env_path="/fake/env", cache_dir=tmp_path)

    assert first == second
    # _create_env is called both times (it owns the cache-hit check itself);
    # what matters is that the resolved path - and therefore the cache key -
    # is stable across calls with the same environment.
    assert build_calls == [first, first]


def test_get_environment_unstaged_fail_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        re_mod, "_find_editable_pip_installs", lambda: {"vine_reduce": "/some/path"}
    )
    monkeypatch.setattr(re_mod, "_local_pip_commits", lambda paths, watch: {"vine_reduce": "HEAD"})
    monkeypatch.setattr(re_mod, "_environment_state_hash", lambda conda_env_path: "abc123")

    with pytest.raises(UnstagedChanges) as exc_info:
        get_environment(conda_env_path="/fake/env", cache_dir=tmp_path, unstaged="fail")

    assert exc_info.value.args[0] == ["vine_reduce"]


def test_get_environment_unstaged_rebuild_forces_create_env(tmp_path, monkeypatch):
    seen_force = []

    def fake_create(env_path, conda_env_path, editable, force=False):
        seen_force.append(force)
        with open(env_path, "wb"):
            pass
        return env_path

    monkeypatch.setattr(
        re_mod, "_find_editable_pip_installs", lambda: {"vine_reduce": "/some/path"}
    )
    monkeypatch.setattr(re_mod, "_local_pip_commits", lambda paths, watch: {"vine_reduce": "HEAD"})
    monkeypatch.setattr(re_mod, "_environment_state_hash", lambda conda_env_path: "abc123")
    monkeypatch.setattr(re_mod, "_create_env", fake_create)

    get_environment(conda_env_path="/fake/env", cache_dir=tmp_path, unstaged="rebuild")

    assert seen_force == [True]


def test_get_environment_raises_without_conda_env_path(tmp_path, monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    with pytest.raises(RuntimeError):
        get_environment(cache_dir=tmp_path)


def test_check_pack_dependencies_passes_when_both_present(monkeypatch):
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: "/usr/bin/poncho_package_create")
    monkeypatch.setattr(
        re_mod.importlib.util, "find_spec", lambda name: object()  # any non-None spec
    )
    _check_pack_dependencies()


def test_check_pack_dependencies_raises_when_ndcctools_missing(monkeypatch):
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(re_mod.importlib.util, "find_spec", lambda name: object())

    with pytest.raises(RuntimeError, match="ndcctools"):
        _check_pack_dependencies()


def test_check_pack_dependencies_raises_when_conda_pack_missing(monkeypatch):
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: "/usr/bin/poncho_package_create")
    monkeypatch.setattr(re_mod.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="conda-pack"):
        _check_pack_dependencies()


def test_get_environment_bails_out_before_reinstalling_editables_when_deps_missing(
    tmp_path, monkeypatch
):
    reinstall_calls = []

    monkeypatch.setattr(
        re_mod, "_find_editable_pip_installs", lambda: {"vine_reduce": "/some/path"}
    )
    monkeypatch.setattr(re_mod, "_local_pip_commits", lambda paths, watch: {"vine_reduce": "abc"})
    monkeypatch.setattr(re_mod, "_environment_state_hash", lambda conda_env_path: "abc123")
    monkeypatch.setattr(re_mod.subprocess, "check_call", lambda *a, **k: reinstall_calls.append(a))
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ndcctools"):
        get_environment(conda_env_path="/fake/env", cache_dir=tmp_path)

    assert reinstall_calls == []
