# vine\_reduce

This file describes the design of the vine\_reduce python module, which generates MapReduce-like workflows for High Energy Physics (HEP). vine\_reduce does not itself execute the workflows: it relies on a **distributor** that manages a distributed high-throughput computation at scale, and on an **executor** that runs individual functions on the remote worker nodes. See the README for installation and a runnable quick start; this file is the design reference.

## HEP Workflows

Typical HEP workflows consists on orthogonal processing functions applied to collision events. Processing functions and collision events are naturally parallel in that one processing function does not affect others, nor the processing of an event affect another. Since the processing of an event is very fast, events are grouped into sets called chunks and processing functions are applied to the chunks. The data is organized into datasets. A dataset consists of a name, metadata and a set of URLs. The URLs identify files that contain the events. Events in a file are numbered from [0, num\_entries) from which chunks can be formed.

### Generating Final Results and Reduction

The result of processing functions is not used as is, but they are merged together with an reducer function. Reducers are associative, distributive, commutative, and generate the same data type as processing function. No two chunks of different datasets should be reduced together, and chunks of a file should not be reduced until all the chunks of that file has been succesfully processed. Chunks never cross file boundaries. In the default case each dataset will generate a single result from final reduction, however some workflows generate several results per dataset.

Once a file's chunks are all successfully processed, its outputs join a single pending pool for the (processor, dataset) pair - chunks are *not* reduced file-by-file first, so results from different files of the same dataset can end up in the same reduction group. Whenever the pool reaches `reduction_size` items, the oldest `reduction_size` are reduced together. When chunk generation is exhausted and fewer than `reduction_size` items remain in the pool - including exactly one, e.g. a single-chunk dataset - that remainder is still reduced/finalized as a smaller group: the pool always drains, it never stalls waiting for input that isn't coming.

Whether a dataset generates one or several final results is decided by an `is_result` function. Before an reduction call for a given (processor, dataset) is submitted, `is_result(num_events, total_time, total_memory)` is invoked, where the three arguments describe the group of not-yet-final results about to be reduced: the number of events they cover, their total execution time, and their total size in memory (see Outcome.resources below). If `is_result` returns True, that group is reduced one final time, `result_postprocess` is applied to the output, and the output becomes a final result that is no longer eligible for further reduction; a new group then starts forming for the same (processor, dataset). The default `is_result` returns True only once all chunks of the dataset have been consumed and reduced, i.e. one final result per dataset.

Reduction functions reduce reduction\_size results per call. reduction\_size should be at least two, but for the edge case of a dataset that consists of a single chunk. In such case, the reduction task functions like a final checkpoint (see next). reduction\_size is managed per processing function x dataset, with a default for all of 10. reduction\_size should be halved (down to a minimum of 2) if the distributor reports resource exhaustion. The default reducer is `f(a, b): a += b; return a`, called as many times as necessary. Care should be taken so that arguments not needed anymore are freed to reduce memory consumption.

## Temporary Results, Checkpoints, and Restart

It is assumed that any intermediate result from processing functions or reductions that are not a final results are temporary. Checkpoints may be generated for a (processor, dataset) once either of two thresholds is crossed: checkpoint\_time (cumulative wall\_time, in seconds) or checkpoint\_size (cumulative memory, in MB). Both are summed per reduction lineage - the chain of reduction calls that produced a given pooled result - not globally per (processor, dataset): each pooled item tracks the wall\_time/memory accumulated since its own lineage was last checkpointed, and crossing a threshold (or becoming a final result) checkpoints that item and resets its accumulator. Either threshold, when set, can independently trigger a checkpoint, and a threshold left as None disables that trigger. Setting checkpoint\_accumulations=True checkpoints every non-final reduction result outright, regardless of the thresholds. vine\_reduce itself does not generate the checkpoint, just manages it and tells the distributor to generate it. Final results are a special kind of checkpoint where their events are not considered for reduction anymore (see `is_result` above). Once a checkpoint is succefully generated, the temporary results related to it can be removed from the cluster via release\_result. Checkpoints that are covered by other checkpoints as results are merged should be also removed from the cluster.

Whether a reduction call will be checkpointed is decided before that reduction is submitted, not after it returns: at submission time, the about-to-be-reduced group's inputs already carry their own since-last-checkpoint wall\_time/memory (summed across the group), and that sum - together with `is_result` - is enough to decide. This means the decision does not (and cannot) account for the wall\_time/memory the reduction call about to be submitted will itself add; that cost is still folded into the resulting item's totals once the call returns, it just is not part of the go/no-go decision. In practice this only shifts a threshold crossing to, at most, one reduction group later than a check made after the fact would. This is purely internal Pipeline bookkeeping (`_submit_reduction` computing `is_checkpoint` on `_ReduceTask`) - it is not part of the Distributor protocol, since a reducer call runs the same way and produces the same file whether or not its result ends up checkpointed.

When checkpoints are generated, information is written to a sqlite database ("the db" from now on). This database is used so that, if the workflow is interrupted, it can be restarted from where it left off: on startup, vine\_reduce reads the checkpoint rows on file for each (processor, dataset) and skips any dataset files they already cover - files under a final checkpoint are done outright, files under a partial checkpoint resume from that checkpoint's pooled result rather than reprocessing their chunks. A checksum of each dataset's definition (name, metadata, files) is stored alongside its checkpoints; if that checksum changes between runs, that dataset's checkpoints are discarded and it restarts from scratch, since there is no way to know its existing checkpoints still correspond to the new definition. The db trades durability for write throughput (`PRAGMA synchronous = OFF`): a crash before a checkpoint's write commits just means that checkpoint never happened and its work is redone on restart - the same cost as any interval that hasn't been checkpointed yet, not new data loss. A checkpoint's insert and its superseded rows' deletes are always committed together in one transaction, so a restart never sees a superseded row deleted without its replacement present, or vice versa.

From the distributor perspective there is a distiction between a workflow result and function outcome. A workflow result is the data the user is interested in producing. A function outcome is a union data type that may be success, failure, or resource exhaustion, and every variant carries a `result_id` (matching the id returned by the `submit` call it answers) and a `resources` dictionary reporting what the task actually used, e.g. `{"cores": ..., "memory_mb": ..., "wall_time_s": ...}`; this is measured by executor\_wrapper/reducer\_wrapper using core python modules where possible (e.g. `resource.getrusage`, `time.monotonic`). The failure case additionally contains the traceback of the processing/reduction function when it failed and should contain the information for debugging. vine\_reduce itself responds to function outcomes and not workflow results. Workflow results should never be read into memory by vine\_reduce, only remotely by the distributor because they may be too large in terms of memory or deserialization time.

## Priorities

All processing functions of the same processor per dataset will have the same priority for execution (the larger the integer number, the better priority). A processor declared earlier will have better priority than a later one. Reduction functions function in the same way, only that they have better priority than any processing function. The purpose of this priority is to finish processing a (processor, dataset) before moving to the next one, but allow overlapping on long tails if there are resources available.

## Chunksize

Management of chunksize is per processor x dataset. Initial chunksizes can be given that applies to all combinations, or per processor, or per dataset. When more than one of these is given for a (processor, dataset) pair, the most specific wins: a per-dataset chunksize overrides a per-processor chunksize, which in turn overrides the global default. Chunksize is dynamic and it will change as performance information is available from the distributor. The most basic chunksize management is to half it (down to a minimum of 1) when the distributor reports that a processing function failed because it exhausted it resources. If no default chunksizes are given, then all the events of a file are chunked together.

## Data Flow

```
┌─ LOCAL  (vine_reduce process) ───────────────────────────────────────────────┐
│                                                                              │
│ input description (file, user given)                                         │
│   │                                                                          │
│   ▼                                                                          │
│ input_to_datasets()                                                          │
│   │ ──► datasets {name: {metadata, files: {url: num_entries}}}               │
│   ▼                                                                          │
│ datasets_to_chunks()   generator, restarted per processor                    │
│   │ ──► Chunk(url, start, stop)                                              │
│   │     throttled by distributor.capacity(), max_chunks_active,              │
│   │     max_chunks_cycle; chunksize halved on resource exhaustion            │
│   ▼                                                                          │
│ is_result(num_events, total_time, total_memory)                              │
│   │ ──► decides: keep reducing, or emit a final result                       │
│   ▼                                                                          │
│ checkpoint logic (checkpoint_time / checkpoint_size thresholds)              │
│   ├──► sqlite db          (progress, checksums, restart state)               │
│   ├──► distributor.release_result(result_id)  (superseded temp results)      │
│   └──► checkpoint_dir/ , results_dir/  (distributor.retrieve() copies files  │
│         here if checkpoint_retrieve / results_retrieve is True)              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                             │
                             │  submit(priority, category, kind,
                             │         executor_wrapper | reducer_wrapper, ...)
                             ▼
        distributor dispatches the call to a worker node
                             ▲
                             │  wait() ──► Outcome(result_id, resources, ...)
                             │
┌─ REMOTE  (worker nodes, one instance per submitted call) ────────────────────┐
│                                                                              │
│ executor_wrapper(chunk, dataset_metadata, distributor_metadata,              │
│                   executor_metadata)                                         │
│   │                                                                          │
│   ▼                                                                          │
│ chunk_to_args(chunk, dataset_metadata, distributor_metadata) ──► args        │
│   │                                                                          │
│   ▼                                                                          │
│ executor(processor, args, dataset_metadata, distributor_metadata,            │
│          executor_metadata)                                                  │
│   │ ──► processor(args) ──► processing result                                │
│   ▼                                                                          │
│ result serialized to a file local to the worker node                         │
│   ──► Outcome{result_id, resources, file | traceback}                        │
│                                                                              │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─      │
│                                                                              │
│ reducer_wrapper(reducer, results, is_final)                                  │
│   │                                                                          │
│   ▼                                                                          │
│ reducer(a, b) ──► reduced result                                             │
│   │                                                                          │
│   ▼  (only if is_result() returned True for this group)                      │
│ result_postprocess(result)                                                   │
│   │                                                                          │
│   ▼                                                                          │
│ result serialized to a file local to the worker node                         │
│   ──► Outcome{result_id, resources, file | traceback}                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

data:     input description:  an arbitrary text file that describes dataset, files and metadata per dataset.
                              User given, or an already-parsed dict of the same shape (see input\_to\_datasets).
function: input\_to\_datasets: converts input description into a dictionary where keys are datasets, and
                              values are dictionaries with metadata and files values. The value of the key
                              files is also a dictionary where keys are urls and values are (at least)
                              num\_entries. This function executes locally with the process running vine\_reduce.
                              User given. The default accepts a dict directly, or loads one from a json file path.
data:     datasets:           As dictionary as described above. If a persistent checksum in the db of this
                              value changes, then that dataset's checkpoints are discarded and it restarts
                              from scratch (see "Temporary Results, Checkpoints, and Restart" above).
                              Generated by vine\_reduce.
generator: datasets\_to\_chunks Generate one by one the chunks from the datasets data. This generator is
                              restarted per processing function. Not all chunks should be generated at once
                              to allow the chunksize to adapt accordingly. A parameter of max\_chunks\_active can be set
                              to limit the number of chunks currently being processed by the distributor
                              (i.e., submitted but not yet released). Also max\_chunks\_cycle sets a limit on
                              how many chunks can be given to the distributor in a single call to
                              datasets\_to\_chunks. The distributor should return the number of chunks it
                              can currently handle from a call to its capacity method; this number is capped
                              by max\_chunks\_active minus chunks currently in flight, and by
                              max\_chunks\_cycle per call.
                              This function executes locally with the process running vine\_reduce.
                              The default is to generate chunks according to the current chunksize for the
                              processor x dataset combination, but a uset can override it.
function: chunk\_to\_args       Converts a Chunk (url, start, stop) into data on which a processor can be
                              applied. It also has a mandatory argument "dataset\_metadata". It has an optional
                              "distributor\_metadata" argument, which may contain
                              "resources": {"cores": ..., "memory": ...} of the resources available to the
                              processor. This function executes remotely per chunk at the worker nodes.
function: executor            Calls the processor on the chunk. It gets as arguments the processor,
                              the result to chunk\_to\_args and the same metadata arguments as
                              chunk\_to\_args, plus an optional "executor\_metadata" dictionary.
                              The default (simple\_executor) is to simply call the processor on the result to
                              chunk\_to\_args, but it can be overriden by the user - see "Executors" below for
                              two alternatives shipped with vine\_reduce.
function: executor\_wrapper   calls chunk\_to\_args and executor as above, and generates the function outcome as
                              needed. This outcome it is what the distributor reports to vine\_reduce thus
                              it should trap any exceptions and captured the traceback as necessary.
                              The executor\_wrapper call is generated by vine\_reduce but executed remotely
                              at the worker nodes.
                              On success, it serializes its result to a file (given as an argument) local
                              to the worker node running the task; this file is entirely maintained by the
                              distributor. The Outcome returned to vine\_reduce carries the path/handle to
                              this file on Success, not the result itself, so vine\_reduce never has to
                              deserialize or hold workflow results in memory.
function: is_result           Decides whether the group of not-yet-final results about to be
                              reduced for a (processor, dataset) should become a final result.
                              Called locally by vine\_reduce with (num\_events, total\_time,
                              total\_memory) before submitting that reduction call, using the
                              resources reported in the Outcomes of the results being merged.
                              This function executes locally with the process running vine\_reduce.
function: result\_postprocess An optional function to apply to the result of reductions that are
                              final results (i.e., where is_result returned True).
                              This function is user defined and executed remotely at the worker nodes.
function: reducer\_wrapper: Calls the reduction function and generates its outcome as needed.
                              If succesful, and after applying result\_postprocess if this is a final
                              result, it writes the result (not the outcome) to a file given as an argument.

## API vine\_reduce <-> distributor

A distributor implements the `Distributor` protocol (`src/vine_reduce/distributor.py`):

```python
result_id = submit(priority, category, kind, executor_wrapper | reducer_wrapper, *args): submit a
    processor or reduction function call. Category is a string that identifies functions of the same
    processing/reduction set (e.g. for logging or scheduling heuristics). kind is "processor" or
    "reducer", letting a distributor apply different resource requests to each. Returns a result_id,
    used later to release_result/retrieve. func is always called as func(dest_file, *args): a
    distributor picks dest_file (or an opaque token standing in for it) and prepends it itself -
    vine_reduce never picks the path, it only ever sees it echoed back on Outcome.file.
outcome = wait(timeout): wait for a result to be available and return its Outcome (RuntimeFailure |
    ResourceExhaustion | Success). On timeout return None. outcome.result_id identifies which submit()
    call this outcome corresponds to.
release_result(result_id): release any resources (e.g. worker-local files) held for result_id.
chunks_wanted = capacity(): number of additional chunks the distributor could handle given the current
    resources.
retrieve(result_id, dest_path): copy/materialize the file for a completed (Success) result_id to
    dest_path, a path local to the vine_reduce process. vine_reduce calls this (instead of assuming
    Outcome.file is directly readable) whenever checkpoint_retrieve or results_retrieve is True - this
    keeps the interface correct for a distributor that doesn't share a filesystem with vine_reduce.
add_file(local_path, remote_path=None): make local_path available, under remote_path (defaulting to
    local_path's basename), wherever every call submitted from now on runs. A no-op for a distributor
    whose workers already share vine_reduce's filesystem.
set_env_var(name, value): set an environment variable for every call submitted from now on.
shutdown(): release whatever resources this distributor owns (worker pools, temp directories, ...).
    Also reachable via `with distributor: ...`.
```

`add_file`/`set_env_var` are called once per entry in `VineReduce.extra_files`/`environment_variables`, at the very start of `compute()`, before any task is submitted - so a caller can hand a processor its supporting files (e.g. a data file read by relative path, an auth token/proxy) and env vars (e.g. `X509_USER_PROXY`) without VineReduce itself needing to know anything distributor-specific.

## Dataclasses

```python
VineReduce:
processors Dict[str, Callable]: Mapping from processor names to processing functions.
input str | dict: pathname to the input description, or an already-parsed dict of that shape.
input_to_datasets Optional[Callable]: Convert input into the dictionary of datasets.
datasets_to_chunks Optional[Callable]: Generate chunks per dataset. Reset per processor.
chunk_to_args Callable: Instantiate chunks.
executor Callable = simple_executor: Call processor on instantiated chunks. See "Executors" below.
reducer Callable = default_reducer: Function to merge to results together.
reduction_size int | dict = 10: Results to reduce together in a single reduction call. Either a plain
                               int, or {"default": int, "processors": {name: int}, "datasets": {name: int}}
                               for per-processor/per-dataset overrides (most specific wins).
is_result Optional[Callable] = None: is_result(num_events, total_time, total_memory) decides whether
                               the output of an reduction call for a (processor, dataset) is a
                               final result, or should keep being reduced with later results.
                               Default: True only once all chunks of the dataset are consumed.
result_postprocess Optional[Callable] = None: Function to apply to results that are final results.
checkpoint_time Optional[float]: Wall_time (seconds), per reduction lineage, that would trigger a checkpoint.
checkpoint_size Optional[float]: Memory (MB), per reduction lineage, that would trigger a checkpoint.
checkpoint_accumulations bool = False: If True, checkpoint every non-final reduction result outright,
                               regardless of checkpoint_time/checkpoint_size.
checkpoint_dir str = "checkpoints": Local directory to write checkpoints.
checkpoint_retrieve bool = True: Whether the distributor should copy the checkpoints to checkpoint_dir.
                                 If False, it is assumed that the distributor has its own permanent storage.
results_dir str = "results": Local directory to write results, with one subdirectory per dataset
                              and, within that, one subdirectory per processor (so multiple
                              processors run over the same dataset don't collide).
results_retrieve bool = True: Whether the distributors should copy results to results_dir.
                              If False, it is assumed that the either the distributor has its own
                              permanent storage, or that result_postprocess copied the result to an
                              appropiate location.
distributor Optional[Distributor]: The distributor to use. Defaults to a LocalDistributor that
                              compute() creates and tears down itself.
chunksize int | dict | None: Target number of events per chunk, same shape as reduction_size.
                              None means one chunk per file. Halved automatically on resource
                              exhaustion, taking effect for chunks not yet generated.
max_chunks_active int = 1000: Cap on chunks in flight (submitted but not yet finished) across all
                              pipelines at once.
max_chunks_cycle int = 100: Cap on new chunks submitted per scheduling cycle, across all pipelines.
db_path Optional[str]: Path to the sqlite checkpoint database; defaults to checkpoint_dir/vine_reduce.db.
extra_files List[str] = []: Local paths made available, under their basename, to every processor/
                            reducer call. Passed to the distributor via add_file() once, at the
                            start of compute().
environment_variables Dict[str, str] = {}: Environment variables set for every processor/reducer
                                           call. Passed to the distributor via set_env_var() once,
                                           at the start of compute().
```

```python
Chunk:
url str: URL from where to get the events.
start int: Inclusive start of the chunk.
stop int: Exclusive end of the chunk.
num_events int: property, stop - start.
```

```python
Outcome: Union of RuntimeFailure, ResourceExhaustion, Success. All variants carry:
  result_id: id returned by the submit() call this outcome corresponds to.
  resources Dict[str, Any]: resources used by the task, e.g.
                            {"cores": ..., "memory_mb": ..., "wall_time_s": ...}.
                            Measured by executor_wrapper/reducer_wrapper using core python
                            modules where possible (e.g. resource.getrusage, time.monotonic).

RuntimeFailure additionally carries:
  traceback str: captured traceback of the processing/reduction function failure.

Success additionally carries:
  file str: the distributor's own handle, local to the worker node, for where
            executor_wrapper/reducer_wrapper serialized its result - a real path for a distributor
            whose workers share vine_reduce's filesystem, or an opaque token otherwise (see
            TaskVineDistributor below). Passed to Distributor.retrieve() to materialize it locally.
```

Worker-side, `executor_wrapper`/`reducer_wrapper` actually return a `RawOutcome` (`status`, `resources`, `file | traceback`) - the distributor-agnostic value produced before a `result_id` exists. A distributor attaches the `result_id` it assigned at `submit()` time via `RawOutcome.to_outcome(result_id)` to produce the `Outcome` it hands back from `wait()`.

## Distributors

Two `Distributor` implementations ship with vine\_reduce.

### LocalDistributor (`src/vine_reduce/local_distributor.py`)

The default when `distributor=` is omitted: runs every processor/reducer call in a local `ProcessPoolExecutor`, for development, testing, and as the minimal reference implementation of the protocol.

- Worker "nodes" are local subprocesses that already share vine\_reduce's filesystem, so `retrieve()` is a plain file copy and `add_file()` is a no-op.
- `func`/`args` are cloudpickled before crossing into the subprocess, so `processor`/`reducer`/etc. may be closures or lambdas, not just module-level callables.
- Priority is best-effort only: a pending call waits in a priority queue until a worker slot is free, but once dispatched to the pool it cannot be preempted by a higher-priority call submitted later.

### TaskVineDistributor (`src/vine_reduce/taskvine_distributor.py`)

Runs vine\_reduce across a real cluster of machines via [TaskVine](https://cctools.readthedocs.io/en/stable/taskvine), instead of local subprocesses.

- **Manager-only, external workers.** The constructor starts a `vine.Manager` and nothing else; `vine_worker` processes, factories, or batch submission are the caller's responsibility, the normal way TaskVine is used. A pre-built `manager` (e.g. a `vine.DaskVine`) can be passed in instead, so vine\_reduce's own map/reduce tasks can share one manager/port/worker pool with a caller's other tasks (e.g. coffea's `dataset_tools.preprocess(scheduler=...)`).
- **File-passing via opaque tokens.** TaskVine workers generally don't share a filesystem with the manager or each other, so a real path can't stand in for `Outcome.file`. Every result becomes a `manager.declare_temp()` file, kept at/near the worker that produced it; `Outcome.file` is an opaque token this class mints (e.g. `"result_7.p"`). When that token later appears inside a later `submit()` call's args (a `reducer_wrapper` `input_files` entry), it's recognized, the underlying `vine.File` is attached as a task input under a fresh sandbox name, and that name is substituted into the args actually sent - so the remote wrapper opens a name that exists in its own sandbox, never the manager-side token. `retrieve()` is the only place bytes are pulled to the manager, via `manager.fetch_file()` + `File.contents()` (binary-safe, unlike `Manager.fetch_file()`'s own return value, which round-trips through a C string and truncates on embedded NUL bytes).
- **Infra-level resource exhaustion is mapped, not just Python-level.** TaskVine's resource monitor (`enable_monitoring(watchdog=True)`) can kill and report a task that overruns its resource allocation - something a plain `ProcessPoolExecutor` can't detect at all. `wait()` trusts the in-process `RawOutcome` only when `task.successful()`; otherwise it maps TaskVine's own result string (`"resource exhaustion"`, `"max wall time"`, `"disk alloc full"` -> `ResourceExhaustion`, anything else -> `RuntimeFailure`), so chunksize/reduction\_size halving is reachable from real cluster failures too.
- **Resources are per-category, not per-task.** `resources_processor`/`resources_reducer` (each an optional `{"cores", "memory_mb", "disk_mb"}` dict) are fixed constructor args, applied via `manager.set_category_resources_max(category, ...)` the first time each distinct category string is seen. `submit()`'s `kind` parameter (`"processor"` or `"reducer"`) selects which of the two applies to a task, since the caller-supplied category string is not a reliable signal on its own.
- `add_file`/`set_env_var` remember what's been added and attach it (`declare_file`/`add_input`, `set_env_var`) to every task submitted from then on. An optional poncho `environment` (see "Executors and remote environments" below) is likewise attached to every task.
- Tested against a real `vine_worker` subprocess (skipped if not on `PATH`), not a fake - see the `taskvine-local-testing` skill and `tests/test_taskvine_distributor.py`.

## VineReduceCoffea

`VineReduceCoffea` (`src/vine_reduce/coffea.py`) is a `VineReduce` specialization for [coffea](https://coffeateam.github.io/coffea/)-based workflows over NanoEvents. It only supplies the coffea-specific pieces; chunking, checkpointing, and restart are inherited unchanged from `VineReduce`:

- `input_to_datasets` defaults to `coffea_input_to_datasets`, which converts the output of coffea's own `preprocess()` (files described by `{"num_entries": ..., "steps": ..., "uuid": ...}`) into vine\_reduce's `{url: num_entries}` shape. Accepts that dict directly, or a path to a json file holding it.
- `chunk_to_args`/`executor` are built in `__post_init__` from `schema`/`mode`/`uproot_options`/`object_path` (`chunk_to_args` opens `NanoEventsFactory.from_root` over `[chunk.start, chunk.stop)`) and from `processor_args` (`executor` calls `processor(events, **processor_args)` then recursively materializes any virtual awkward arrays in the result before it's pickled).
- `reducer` defaults to `default_reducer`, a coffea-flavored accumulator (handles `Addable`, `MutableSet`, and recursive `MutableMapping` merging), since base VineReduce's `a += b` default can't merge the dict-of-histograms shape coffea processors typically return.

## Executors and remote environments

`executor` (`src/vine_reduce/executor.py`) controls how a processor call actually runs at the execution site, once a distributor has placed it on a worker: `simple_executor` (default) calls `processor(args)` directly; `cloudpickle_executor` isolates the call in its own subprocess so a crash or memory leak in `processor` doesn't take down the worker task; `dask_executor` is for a `processor` that returns a dask-delayed object (or array/dataframe) and computes it at the execution site. See the README's "Executors" section for the full rundown.

Workers need nothing beyond a distributor pre-installed. `get_environment()` (`src/vine_reduce/remote_environment.py`) packs the calling conda environment (`$CONDA_PREFIX` by default) into a relocatable [poncho](https://cctools.readthedocs.io/en/stable/poncho) tarball via `poncho_package_create`, for `TaskVineDistributor(environment=...)` to ship and activate on every worker. Builds are cached on disk and rebuilt automatically whenever a locally-editable package (vine\_reduce itself, by default, or anything else named via `pip_editable`) has uncommitted changes, so a tarball never silently ships stale code. See the README's "Packaging an environment for remote workers".
