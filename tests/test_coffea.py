from __future__ import annotations

import pytest

from vine_reduce.coffea import VineReduceCoffea, coffea_input_to_datasets, default_reducer


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
    assert vr.executor is not None
    assert vr.reducer is default_reducer
    assert vr.input_to_datasets is coffea_input_to_datasets


def test_vine_reduce_coffea_executor_materializes_result():
    vr = VineReduceCoffea(processors={"p": lambda events: {"count": len(events)}}, input={})

    def processor(events):
        return {"count": len(events)}

    result = vr.executor(processor, [1, 2, 3], {})
    assert result == {"count": 3}
