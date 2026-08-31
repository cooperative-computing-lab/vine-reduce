from __future__ import annotations

import json

import pytest

from vine_reduce.coffea import (
    CoffeaExecutor,
    VineReduceCoffea,
    _checksum_fileset,
    coffea_input_to_datasets,
    default_reducer,
)
from vine_reduce.types import Chunk


def test_default_reducer_adds_plain_addables():
    assert default_reducer(1, 2) == 3


def test_default_reducer_merges_mappings_recursively():
    a = {"x": 1, "shared": {"a": 1}}
    b = {"y": 2, "shared": {"b": 2}}
    result = default_reducer(a, b)
    assert result == {"x": 1, "y": 2, "shared": {"a": 1, "b": 2}}


def test_default_reducer_unions_sets():
    assert default_reducer({1, 2}, {2, 3}) == {1, 2, 3}


def test_default_reducer_rejects_incompatible_mapping_types():
    class OtherDict(dict):
        pass

    with pytest.raises(ValueError):
        default_reducer(OtherDict(), {})


def test_default_reducer_rejects_incompatible_types():
    with pytest.raises(ValueError):
        default_reducer(1, {"a": 1})


def test_coffea_input_to_datasets_converts_file_specs():
    preprocessed = {
        "signal": {
            "metadata": {"xsec": 1.0},
            "files": {
                "a.root": {"object_path": "Events", "num_entries": 100, "uuid": "abc"},
                "b.root": {"object_path": "Events", "num_entries": 50, "uuid": "def"},
            },
        },
        "background": {
            "files": {"c.root": {"object_path": "Events", "num_entries": 10}},
        },
    }
    datasets = coffea_input_to_datasets(preprocessed)
    assert datasets == {
        "signal": {"metadata": {"xsec": 1.0}, "files": {"a.root": 100, "b.root": 50}},
        "background": {"metadata": {}, "files": {"c.root": 10}},
    }


def test_coffea_input_to_datasets_reads_json_file(tmp_path):
    import json

    preprocessed = {"ds": {"files": {"a.root": {"num_entries": 5}}}}
    path = tmp_path / "preprocessed.json"
    path.write_text(json.dumps(preprocessed))

    datasets = coffea_input_to_datasets(str(path))
    assert datasets == {"ds": {"metadata": {}, "files": {"a.root": 5}}}


def test_vine_reduce_coffea_wires_chunk_to_args_and_executor():
    vr = VineReduceCoffea(processors={"p": lambda events: events}, input={})

    # chunk_to_args and executor are built in __post_init__ from schema/mode/etc,
    # so they must be present and distinct from the base VineReduce defaults.
    assert vr.chunk_to_args is not None
    assert isinstance(vr.executor, CoffeaExecutor)
    assert vr.reducer is default_reducer
    assert vr.input_to_datasets is coffea_input_to_datasets


def test_chunk_to_args_defaults_steps_per_file_to_one(monkeypatch):
    vr = VineReduceCoffea(processors={"p": lambda events: events}, input={})

    captured = {}

    def fake_from_root(*args, **kwargs):
        captured.update(kwargs)

        class _Fake:
            def events(self):
                return "events"

        return _Fake()

    monkeypatch.setattr("coffea.nanoevents.NanoEventsFactory.from_root", fake_from_root)

    chunk = Chunk(url="a.root", start=0, stop=10)
    vr.chunk_to_args(chunk, {}, distributor_metadata=None)
    assert captured["steps_per_file"] == 1


def test_chunk_to_args_uses_distributor_cores_as_steps_per_file(monkeypatch):
    vr = VineReduceCoffea(processors={"p": lambda events: events}, input={})

    captured = {}

    def fake_from_root(*args, **kwargs):
        captured.update(kwargs)

        class _Fake:
            def events(self):
                return "events"

        return _Fake()

    monkeypatch.setattr("coffea.nanoevents.NanoEventsFactory.from_root", fake_from_root)

    chunk = Chunk(url="a.root", start=0, stop=10)
    vr.chunk_to_args(chunk, {}, distributor_metadata={"cores": 4})
    assert captured["steps_per_file"] == 4


def test_vine_reduce_coffea_executor_materializes_result():
    vr = VineReduceCoffea(processors={"p": lambda events: {"count": len(events)}}, input={})

    def processor(events):
        return {"count": len(events)}

    result = vr.executor.submit(processor, [1, 2, 3], dataset_metadata={}).result()
    assert result == {"count": 3}


def test_vine_reduce_coffea_executor_forwards_processor_args():
    vr = VineReduceCoffea(
        processors={"p": lambda events: events}, input={}, processor_args={"k": 2}
    )

    def processor(events, k):
        return len(events) * k

    result = vr.executor.submit(processor, [1, 2]).result()
    assert result == 4


def test_coffea_input_to_datasets_raises_on_missing_num_entries():
    preprocessed = {"ds": {"files": {"a.root": {"object_path": "Events", "num_entries": None}}}}
    with pytest.raises(ValueError, match="preprocess_cache"):
        coffea_input_to_datasets(preprocessed)


def test_coffea_input_to_datasets_raises_on_absent_num_entries_key():
    preprocessed = {"ds": {"files": {"a.root": {"object_path": "Events"}}}}
    with pytest.raises(ValueError, match="preprocess_cache"):
        coffea_input_to_datasets(preprocessed)


def test_coffea_input_to_datasets_raises_on_bare_file_spec():
    preprocessed = {"ds": {"files": {"a.root": "Events"}}}
    with pytest.raises(ValueError, match="preprocess_cache"):
        coffea_input_to_datasets(preprocessed)


def test_preprocess_cache_hit_skips_preprocess(tmp_path, monkeypatch):
    fileset = {"ds": {"files": {"a.root": "Events"}}}
    cache_file = tmp_path / "cache.jsonl"
    cached_result = {"ds": {"files": {"a.root": {"num_entries": 100}}}}
    checksum = _checksum_fileset(fileset)
    with open(cache_file, "w") as f:
        f.write(json.dumps({"checksum": checksum}) + "\n")
        f.write(json.dumps(cached_result) + "\n")

    def _boom(*args, **kwargs):
        raise AssertionError("preprocess should not be called on a cache hit")

    monkeypatch.setattr("coffea.dataset_tools.preprocess", _boom)

    result = VineReduceCoffea.preprocess_cache(fileset, cache_file=cache_file)
    assert result == cached_result


def test_preprocess_cache_miss_on_checksum_change(tmp_path, monkeypatch):
    fileset = {"ds": {"files": {"a.root": "Events"}}}
    cache_file = tmp_path / "cache.jsonl"
    with open(cache_file, "w") as f:
        f.write(json.dumps({"checksum": "stale"}) + "\n")
        f.write(json.dumps({"ds": {"files": {"a.root": {"num_entries": 1}}}}) + "\n")

    fresh_available = {"ds": {"files": {"a.root": {"num_entries": 100}}}}
    calls = []

    def fake_preprocess(passed_fileset, **kwargs):
        calls.append((passed_fileset, kwargs))
        return fresh_available, {"ds": {"files": {}}}

    monkeypatch.setattr("coffea.dataset_tools.preprocess", fake_preprocess)

    result = VineReduceCoffea.preprocess_cache(fileset, cache_file=cache_file, step_size=1000)
    assert result == fresh_available
    assert len(calls) == 1
    assert calls[0][0] == fileset
    assert calls[0][1]["step_size"] == 1000

    with open(cache_file) as f:
        header = json.loads(f.readline())
        cached = json.loads(f.readline())
    assert cached == fresh_available
    assert header["checksum"] != "stale"


@pytest.mark.parametrize(
    "contents",
    [
        "",
        '{"checksum": "abc"}\n',
        "not json\n",
        '{"checksum": "abc"}\nnot json either\n',
    ],
)
def test_preprocess_cache_recomputes_on_corrupt_cache(tmp_path, monkeypatch, contents):
    fileset = {"ds": {"files": {"a.root": "Events"}}}
    cache_file = tmp_path / "cache.jsonl"
    cache_file.write_text(contents)

    fresh_available = {"ds": {"files": {"a.root": {"num_entries": 42}}}}
    monkeypatch.setattr("coffea.dataset_tools.preprocess", lambda *a, **k: (fresh_available, {}))

    result = VineReduceCoffea.preprocess_cache(fileset, cache_file=cache_file)
    assert result == fresh_available

    with open(cache_file) as f:
        header = json.loads(f.readline())
        cached = json.loads(f.readline())
    assert cached == fresh_available
    assert "checksum" in header


def test_preprocess_cache_computes_on_missing_file(tmp_path, monkeypatch):
    fileset = {"ds": {"files": {"a.root": "Events"}}}
    cache_file = tmp_path / "does_not_exist.jsonl"
    fresh_available = {"ds": {"files": {"a.root": {"num_entries": 7}}}}
    monkeypatch.setattr("coffea.dataset_tools.preprocess", lambda *a, **k: (fresh_available, {}))

    result = VineReduceCoffea.preprocess_cache(fileset, cache_file=cache_file)
    assert result == fresh_available
    assert cache_file.exists()
