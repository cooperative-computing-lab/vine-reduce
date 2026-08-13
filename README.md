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

## HEP / coffea workflows

`vine_reduce.VineReduceCoffea` is a specialization for
[coffea](https://coffeateam.github.io/coffea/)-based analyses: it supplies
NanoEvents-reading, awkward-array materialization, and coffea-style
accumulator merging, while chunking, checkpointing, and restart are
inherited unchanged from `VineReduce`. See `src/vine_reduce/coffea.py`.

## License

This project is licensed under the Apache License 2.0 - see the
[LICENSE](LICENSE) file for details.
