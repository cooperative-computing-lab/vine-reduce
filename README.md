# VineReduce

A dynamic MapReduce framework for data processing, built on top of
[TaskVine](https://cctools.readthedocs.io/en/latest/taskvine/).

For each `(processor, dataset)` pair, `VineReduce` splits every file in the
dataset into chunks, runs your processor over each chunk remotely (the "map"
step), then repeatedly folds pooled processor outputs together with a
reducer (the "reduce" step) until one final result covers the whole
dataset. Progress is checkpointed along the way, so an interrupted run can
resume without redoing finished work. See [PLAN.md](PLAN.md) for the full
design.

## Installation

This project requires Python 3.13+ and is managed with
[pixi](https://pixi.sh/).

```bash
# Clone the repository
git clone https://github.com/cooperative-computing-lab/vine_reduce.git
cd vine_reduce

# Install the default environment (runtime dependencies only)
pixi install

# Or install the dev environment (adds pytest, black, flake8, pyright)
pixi install -e dev
```

All commands should be run through pixi so they pick up the managed
environment, e.g. `pixi run python your_script.py`.

## Quick Start

`examples/quick_start/quick_start.py` is a self-contained, runnable example:
it generates some toy binary data, starts a `TaskVineDistributor` with a
single local worker (no cluster or separate `vine_worker` process needed),
runs three processors over two datasets, and checks the results for
consistency.

```bash
cd examples/quick_start
pixi run python quick_start.py
```

Reading through that file top-to-bottom (it's heavily commented) is the
fastest way to see how the pieces fit together: `build_datasets` describes
the input shape `VineReduce` expects, `numbers_chunk_to_args` turns a
`Chunk` into processor arguments, and `main()` wires a `TaskVineDistributor`
and `VineReduce` together and calls `compute()`.

The shape of a minimal call looks like this:

```python
from vine_reduce import VineReduce

vr = VineReduce(
    processors={"my_processor": my_processor_fn},
    input=datasets,  # {name: {"metadata": {...}, "files": {path: num_entries}}}
    chunk_to_args=my_chunk_to_args,
    chunksize=10_000,
    results_dir="results",
    checkpoint_dir="checkpoints",
    # distributor defaults to a local ProcessPoolExecutor-backed
    # LocalDistributor if omitted; pass a TaskVineDistributor to run on a
    # real TaskVine cluster instead.
)
vr.compute()
```

## Executors

`chunk_to_args`'s output for a chunk becomes the argument each processor
call runs remotely on. The `executor` argument to `VineReduce` controls how
that call actually runs, at the execution site:

- `simple_executor` (default) — calls `processor(args)` directly.
- `cloudpickle_executor` — runs `processor(args)` in its own subprocess, so a
  crash or memory leak in `processor` doesn't take down the worker task
  itself. Supports closures and lambdas as `processor`, unlike the stdlib
  `pickle` a plain `ProcessPoolExecutor` would require.
- `dask_executor` — for a `processor` that returns a dask-delayed object (or
  dask array/dataframe) rather than a plain value; computes it at the
  execution site using one subprocess per core allocated to the task. `dask`
  is not a `vine_reduce` dependency and must already be installed wherever
  this executor runs.

All three live in `src/vine_reduce/executor.py`.

## HEP / coffea workflows

`vine_reduce.VineReduceCoffea` is a specialization for
[coffea](https://coffeateam.github.io/coffea/)-based analyses: it supplies
NanoEvents-reading, awkward-array materialization, and coffea-style
accumulator merging, while chunking, checkpointing, and restart are
inherited unchanged from `VineReduce`. See `src/vine_reduce/coffea.py`.

`examples/cortado/vr_cortado.py` is a runnable example built on it,
adapted from the ["cortado"
example](https://github.com/cooperative-computing-lab/dynamic_data_reduction/tree/main/examples/cortado)
in `dynamic_data_reduction`, the project this one's dynamic map-reduce loop
descends from: it generates synthetic NanoAOD-like ROOT files for two
datasets, skims each down to events with at least four leptons, and merges
the surviving events per dataset with a custom awkward-array-concatenating
reducer.

```bash
cd examples/cortado
pixi run python vr_cortado.py
```

## Production use: ttbarEFT

[`TopEFT/ttbarEFT`](https://github.com/TopEFT/ttbarEFT) is a CMS
top-quark EFT search that runs its analysis stage through `vine_reduce`
on top of TaskVine, distributing histogram-filling processors over an
HTCondor pool.
[`examples/ttBar/run_processor_with_vr.py`](examples/ttBar/run_processor_with_vr.py)
shows how that integration looked in practice: driving a `ttbarEFT`
`AnalysisProcessor` per lepton channel through `vine_reduce`. It predates
the current `VineReduceCoffea`/`TaskVineDistributor` API described above
(it was written against an earlier `vine_reduce` release), so treat it as
a reference for how a full physics analysis wires up channels,
Wilson-coefficient/histogram selection, and X509 proxy handling around
`vine_reduce`, not as a runnable script against the current API.

## Development

```bash
pixi run -e dev pytest tests/ -v   # run tests
pixi run -e dev black .            # format
pixi run -e dev flake8             # lint
```

CI (GitHub Actions and a mirrored GitLab CI pipeline) runs all three on
every push and pull request.

## License

This project is licensed under the Apache License 2.0 - see the
[LICENSE](LICENSE) file for details.
