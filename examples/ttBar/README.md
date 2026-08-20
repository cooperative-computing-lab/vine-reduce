# ttBar-EFT example

`run_processor_with_vr.py` runs a coffea processor (`analysis_processor.py`,
not included here - supply your own via `--processor`) over a ttbar-EFT-style
sample set, distributed across a TaskVine cluster via `vine_reduce`.

## Running

```
pixi run -e dev python run_processor_with_vr.py samples.json --port 9123-9130 [other options]
```

This script only talks to a real cluster: it opens a `vine.Manager` on
`--port` and waits for independently-launched `vine_worker` processes (or a
batch-system submission, e.g. `condor_submit_workers`) to connect - it does
not spawn any workers itself. See `examples/quick_start` and
`examples/cortado` for the local, self-contained (`vine.Factory`-based)
pattern instead, if you just want to try `vine_reduce` out without a cluster.

## Running without TaskVine (small samples, no cluster)

`run_processor_with_vr.py` has no local/iterative fallback built in. To run
`analysis_processor.py` directly, without `vine_reduce` or TaskVine at all -
useful for a quick check against a handful of files - use coffea's own
executors directly:

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

runner = processor.Runner(
    processor.IterativeExecutor(),
    schema=NanoAODSchema,
    chunksize=100_000,
)
hists = runner(fileset=my_fileset, processor_instance=my_processor, treename="Events")
```

`processor.FuturesExecutor` (a local process/thread pool) is coffea's other
built-in option if `IterativeExecutor` is too slow for the sample size at
hand, still with no cluster involved. Neither of these was kept as a
built-in flag on `run_processor_with_vr.py` - reach for `vine_reduce`
(this script) once the workload actually needs distributing across a
cluster.
