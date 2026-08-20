# vine\_reduce

This file describes a building plan for the vine reduce python module to generate MapReduce-like workflows for High Energy Physics (HEP). vine\_reduce itself does not itself execute the workflows, but relies on distributors that manage a distributed high-throughput computation at scale, and on executors that run individual functions on the remote worker nodes.

## HEP Workflows

Typical HEP workflows consists on orthogonal processing functions applied to collision events. Processing functions and collision events are naturally parallel in that one processing function does not affect others, nor the processing of an event affect another. Since the processing of an event is very fast, events are grouped into sets called chunks and processing functions are applied to the chunks. The data is organized into datasets. A dataset consists of a name, metadata and a set of URLs. The URLs identify files that contain the events. Events in a file are numbered from [0, num\_entries) from which chunks can be formed. 

### Generating Final Results and Reduction

The result of processing functions is not used as is, but they are merged together with an reducer function. Reducers are associative, distributive, commutative, and generate the same data type as processing function. No two chunks of different datasets should be reduced together, and chunks of a file should not be reduced until all the chunks of that file has been succesfully processed. Chunks never cross file boundaries. In the default case each dataset will generate a single result from final reduction, however some workflows generate several results per dataset.

Whether a dataset generates one or several final results is decided by an `is_result` function. Before an reduction call for a given (processor, dataset) is submitted, `is_result(num_events, total_time, total_memory)` is invoked, where the three arguments describe the group of not-yet-final results about to be reduced: the number of events they cover, their total execution time, and their total size in memory (see Outcome.resources below). If `is_result` returns True, that group is reduced one final time, `result_postprocess` is applied to the output, and the output becomes a final result that is no longer eligible for further reduction; a new group then starts forming for the same (processor, dataset). The default `is_result` returns True only once all chunks of the dataset have been consumed and reduced, i.e. one final result per dataset.

Reduction functions reduce reduction\_size results per call. reduction\_size should be at least two, but for the edge case of a dataset that consists of a single chunk. In such case, the reduction task functions like a final checkpoint (see next). reduction\_size is managed per processing function x dataset, with a default for all of 10. reduction\_size should be halved if the distributor reports resource exhaustion. The default reducer is `f(a, b): a += b; return a`, called as many times as necessary. Care should be taken so that arguments not needed anymore are freed to reduce memory consumption.

## Temporary Results, Checkpoints, and Results Log

It is assumed that any intermediate result from processing functions or reductions that are not a final results are temporary. Checkpoints may be generated for a (processor, dataset) once either of two thresholds is crossed: checkpoint\_time (cumulative wall\_time, in seconds) or checkpoint\_size (cumulative memory, in MB). Both are summed from the `resources` field of the Outcomes (see below) of results not yet covered by a checkpoint; either threshold, when set, can independently trigger a checkpoint, and a threshold left as None disables that trigger. vine\_reduce itself does not generate the checkpoint, just manages it and tells the distributor to generate it. Final results are a special kind of checkpoint where their events are not considered for reduction anymore (see `is_result` above). Once a checkpoint is succefully generated, the temporary results related to it can be removed from the cluster via free\_result. Checkpoints that are covered by other checkpoints as results are merged should be also removed from the cluster.

When checkpoints are generated, information to a sqlite database ("the db from now on") should be updated. This database should be used so that, if the workflow is interrupted, it can be restarted from where it left off.

From the distributor perspective there is a distiction between a workflow result and function outcome. A workflow result is the data the user is interested in producing. A function outcome is a union data type that may be success, failure, or resource exhaustion, and every variant carries a `result_id` (matching the id returned by the `submit` call it answers) and a `resources` dictionary reporting what the task actually used, e.g. `{"cores": ..., "memory_mb": ..., "wall_time_s": ...}`; this is measured by executor\_wrapper/reducer\_wrapper using core python modules where possible (e.g. `resource.getrusage`, `time.monotonic`). The failure case additionally contains the traceback of the processing/reduction function when it failed and should contain the information for debugging. vine\_reduce itself responds to function outcomes and not workflow results. Workflow results should never be read into memory by vine\_reduce, only remotely by the distributor because they may be too large in terms of memory or deserialization time.

## Priorities

All processing functions of the same processor per dataset will have the same priority for execution (the larger the integer number, the better priority). A processor declared earlier will have better priority than a later one. Reduction functions function in the same way, only that they have better priority than any processing function. The purpose of this priority is to finish processing a (processor, dataset) before moving to the next one, but allow overlapping on long tails if there are resources available.

## Chunksize

Management of chunksize is per processor x dataset. Initial chunksizes can be given that applies to all combinations, or per processor, or per dataset. When more than one of these is given for a (processor, dataset) pair, the most specific wins: a per-dataset chunksize overrides a per-processor chunksize, which in turn overrides the global default. Chunksize is dynamic and it will change as performance information is available from the distributor. The most basic chunksize management is to half it when the distributor reports that a processing function failed because it exhausted it resources. If no default chunksizes are given, then all the events of a file are chunked together.

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
│   │     throttled by distributor.hungry(), max_chunks_active,                │
│   │     max_chunks_cycle; chunksize halved on resource exhaustion            │
│   ▼                                                                          │
│ is_result(num_events, total_time, total_memory)                              │
│   │ ──► decides: keep reducing, or emit a final result                       │
│   ▼                                                                          │
│ checkpoint logic (checkpoint_time / checkpoint_size thresholds)              │
│   ├──► sqlite db          (progress, checksums, restart state)               │
│   ├──► distributor.free_result(result_id)  (superseded temp results)         │
│   └──► checkpoint_dir/ , results_dir/  (distributor copies files here        │
│         if checkpoint_retrieve / results_retrieve is True)                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                             │
                             │  submit(priority, category, executor_wrapper
                             │         | reducer_wrapper, ...)
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
                              User given.
function: input\_to\_datasets: converts input description into a dictionary where keys are datasets, and
                              values are dictionaries with metadata and files values. The value of the key
                              files is also a dictionary where keys are urls and values are (at least)
                              num\_entries. This function executes locally with the process running vine\_reduce.
                              User given. The default is to load the input description as json file.
data:     datasets:           As dictionary as described above. If a persistent checksum in the db of this
                              value changes, then the whole workflow should be restarted.
                              Generated by vine\_reduce.
generator: datasets\_to\_chunks Generate one by one the chunks from the datasets data. This generator is
                              restarted per processing function. Not all chunks should be generated at once
                              to allow the chunksize to adapt accordingly. A parameter of max\_chunks\_active can be set
                              to limit the number of chunks currently being processed by the distributor
                              (i.e., submitted but not yet freed). Also max\_chunks\_cycle sets a limit on
                              how many chunks can be given to the distributor in a single call to
                              datasets\_to\_chunks. The distributor should return the number of chunks it
                              can currently handle from a call to its hungry method; this number is capped
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
                              The default is to simply call the processor on the result to chunk\_to\_args,
                              but it can be overriden by the user.
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

A distributor has these methods:

```python
result_id = submit(priority, category, kind, executor_wrapper | reducer\_wrapper, *args): submit a processor or reduction function call. Category is a string that identifies functions of the same processing/reduction set. kind is "processor" or "reducer", letting a distributor apply different resource requests to each. Returns a result_id, used later to free_result.
outcome = wait(timeout): wait for a result to be available and return its Outcome (RuntimeFailure | ResourceExhaustion | Success). On timeout return None. outcome.result_id identifies which submit() call this outcome corresponds to.
free_result(result_id): remove resources associated with result_id
chunks_wanted = hungry(): number of additional chunks the distributor could handle given the current resources.
add_file(local_path): make local_path available, under its basename, wherever every call submitted
                       from now on runs.
set_env_var(name, value): set an environment variable for every call submitted from now on.
```


## Dataclasses

```python
VineReduce:
processors Dict[str, Callable: Mapping from processor names to processing functions
input str: pathname to the input description
input_to_datasets Optional[Callable]: Convert input into the dictionary of datasets
datasets_to_chunks Optional[Generator[Chunk]]: Generate chunks per dataset. Reset per processor.
chunk_to_args Callable: Instantiate chunks.
executor Optional[Callable]: Call processor on instantiated chunks
reducer Optional[Callable]: Function to merge to results together.
reduction_size int = 10: Results to reduce together in a single reduction call. 
is_result Optional[Callable] = None: is_result(num_events, total_time, total_memory) decides whether
                               the output of an reduction call for a (processor, dataset) is a
                               final result, or should keep being reduced with later results.
                               Default: True only once all chunks of the dataset are consumed.
result_postprocess Callable: Function to apply to results that are final results.
checkpoint_time Optional[int]: Total wall_time (seconds) of results in an reduction not yet
                               covered by a checkpoint that would trigger a checkpoint.
checkpoint_size Optional[int]: Total memory (MB) of results in an reduction not yet covered
                               by a checkpoint that would trigger a checkpoint.
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
distributor Optional[Distributor]: The distributor to use. The default distributor is a local default.
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
  file str: path to the file, local to the worker node, where executor_wrapper/reducer_wrapper
            serialized its result.
```

## Implementation Clarifications

These resolve ambiguities in the plan above, decided during implementation:

1. **Restart support**: Full restart-from-db is implemented. On startup, vine\_reduce reads
   checkpoint rows for each (processor, dataset) from the sqlite db and skips files already
   covered by a checkpoint. A per-dataset checksum is stored in the db; if the input dataset's
   checksum changes, that dataset's checkpoints are discarded and it restarts from scratch.

2. **Distributor API gains a fifth method**, `retrieve(result_id, dest_path)`, beyond the
   submit/wait/free_result/hungry described above:
   ```python
   retrieve(result_id, dest_path): copy/materialize the file for a completed result_id to
                                    dest_path, a path local to the vine_reduce process.
   ```
   vine\_reduce calls this (instead of assuming `Outcome.file` is directly readable) whenever
   `checkpoint_retrieve` or `results_retrieve` is True. This keeps the interface correct for
   distributors that don't share a filesystem with vine\_reduce, even though the local
   distributor's worker processes do share one.

3. **Reduction pooling across files**: once a file's chunks are all successfully processed, its
   chunk results join a single pending pool for the (processor, dataset) pair (not reduced
   file-by-file first). Whenever the pool reaches `reduction_size` items, the oldest
   `reduction_size` are reduced together. When chunk generation for that pair is exhausted and
   fewer than `reduction_size` items remain in the pool (including exactly one, e.g. a
   single-chunk dataset), that remainder is still reduced/finalized as a smaller group — the
   pool is always drained, it never stalls waiting for more input that isn't coming.

4. **Checkpointing granularity**: `checkpoint_time`/`checkpoint_size` accumulate per reduction
   lineage (the chain of reduction calls that produced a given pooled result), not globally per
   (processor, dataset). Each pooled item tracks the file URLs it covers and the wall_time/memory
   accumulated since its lineage was last checkpointed; crossing a threshold (or becoming a final
   result) persists that item as a checkpoint row and resets its accumulator. When a new
   checkpoint's inputs were themselves checkpoints, the superseded rows (and their files) are
   deleted, per "checkpoints that are covered by other checkpoints ... should also be removed."

5. **`VineReduce.input` accepts a dict, not just a file path**: the field type is
   `str | dict[str, Any]`, and `default_input_to_datasets` returns the dict as-is when it isn't a
   string, instead of always treating it as a json path. This lets specializations pass an
   already-in-memory dataset description straight through, without a serialize/reload round trip.

6. **`VineReduceCoffea(VineReduce)`** (`src/vine_reduce/coffea.py`) specializes vine\_reduce for
   coffea workflows over NanoEvents. It only supplies the coffea-specific pieces; chunking,
   checkpointing, and restart are inherited unchanged:
   - `input_to_datasets` defaults to `coffea_input_to_datasets`, which converts the output of
     coffea's own `preprocess()` (files described by `{"num_entries": ..., "steps": ..., "uuid":
     ...}`) into vine\_reduce's `{url: num_entries}` shape. Accepts that dict directly, or a path
     to a json file holding it.
   - `chunk_to_args`/`executor` are not dataclass fields the caller sets directly; `__post_init__`
     builds them from `schema`/`mode`/`uproot_options`/`object_path` (chunk_to_args opens
     `NanoEventsFactory.from_root` over `[chunk.start, chunk.stop)`) and from `processor_args`
     (executor calls `processor(events, **processor_args)` then recursively materializes any
     virtual awkward arrays in the result before it's pickled).
   - `reducer` defaults to `default_reducer`, a coffea-flavored accumulator (handles addables,
     `MutableSet`, and recursive `MutableMapping` merging) since base VineReduce's `a += b` default
     can't merge the dict-shaped outputs coffea processors typically return.
   - Deliberately *not* ported from the older taskvine-specific prototype this was modeled after:
     flexible processor shapes (list/dict/`.process`-object) - callers pass `dict[str, Callable]`
     directly, matching base VineReduce; and preprocessing-time `step_size`/`task_splitter` chunk
     splitting - superseded by VineReduce's existing `chunksize` + resource-exhaustion halving.

7. **`TaskVineDistributor`** (`src/vine_reduce/taskvine_distributor.py`) implements the
   `Distributor` protocol on top of `ndcctools.taskvine`, for running across a real cluster
   instead of local subprocesses. Modeled after an older taskvine-specific prototype (a
   `DynamicDataReduction`/`vine.Manager` class that did its own chunking, checkpointing, and
   reduction bookkeeping directly against taskvine), but only the distributor-level mechanics
   were kept - chunking/checkpointing/reduction-pooling stay in pipeline.py, unchanged. Resolved
   via `AskUserQuestion` since these are genuine architecture decisions, not derivable from
   PLAN.md or the prototype alone:
   - **Manager-only, external workers**: the constructor starts a `vine.Manager` and nothing
     else; `vine_worker` processes/factories/batch submission are the caller's responsibility,
     the normal way TaskVine is used. (Rejected: auto-spawning local workers like
     `LocalDistributor` does, since that's extra process-management complexity for what's meant
     to be the "real cluster" distributor, not a self-contained dev/test one.)
   - **File-passing via opaque tokens**: `distributor.py`'s protocol passes result "files" around
     as plain strings that `reducer_wrapper` opens directly - fine for `LocalDistributor` since
     its subprocesses share vine_reduce's filesystem, but TaskVine workers generally don't share
     one with the manager or each other. Every result is a `manager.declare_temp()` file (kept
     at/near the worker, never pulled back automatically); `Outcome.file` is an opaque token this
     class mints (e.g. `"result_7.p"`), not a real path. When that token later appears inside a
     later `submit()` call's args (`reducer_wrapper`'s `input_files` list), `_remap_files`
     recognizes it, attaches the underlying `vine.File` as a task input under a fresh sandbox
     name, and substitutes that name into the args actually sent - so `reducer_wrapper` opens a
     name that exists in its own sandbox, never the manager-side token. `retrieve()` is the only
     place bytes are actually pulled to the manager, via `manager.fetch_file()` +
     `File.contents()` (confirmed empirically to be binary-safe; `Manager.fetch_file()`'s own
     return value is not - it round-trips through a C string and truncates on embedded NUL
     bytes). (Rejected: extending the `Distributor` protocol with an explicit
     "as-input" method, to keep `distributor.py`/`pipeline.py` and `LocalDistributor` untouched.)
   - **Infra-level resource exhaustion is mapped, not just Python-level**: TaskVine's resource
     monitor (`enable_monitoring(watchdog=True)`, on unconditionally) can kill and report a task
     that overruns its resource allocation - something `LocalDistributor`'s plain
     `ProcessPoolExecutor` can't detect at all. `wait()` checks `task.successful()` first; only
     then does it trust the `RawOutcome` that `executor_wrapper`/`reducer_wrapper` returned
     (their own in-process exception handling). Otherwise it maps TaskVine's own `task.result`
     string (`"resource exhaustion"`, `"max wall time"`, `"disk alloc full"` -> `ResourceExhaustion`;
     anything else -> `RuntimeFailure`) so pipeline.py's chunksize/reduction_size halving is
     reachable from real cluster failures, not just simulated `MemoryError`.
   - Resource requests are two fixed per-distributor-instance constructor args,
     `resources_processor`/`resources_reducer` (each an optional `{"cores": ..., "memory_mb": ...,
     "disk_mb": ...}` dict; any key left out is simply not passed through, so TaskVine falls back
     to its own default for that resource). These are applied via
     `manager.set_category_resources_max(category, rmd)` - a category-level TaskVine setting, not
     a per-task one - the first time each distinct category string is seen in `submit()`, not on
     every task. `task.set_category(category)` uses the caller-supplied category string as-is
     (e.g. `"{processor_name}:{dataset_name}:process"` from pipeline.py); `kind` (see below), not
     the category string, selects which of `resources_processor`/`resources_reducer` gets applied
     to it. No per-category autolabeling/tuning (e.g. dynmapred's `"min waste"` category modes).
     An optional poncho `environment` is likewise a single fixed constructor arg applied to every
     task.
   - **Constructor takes an optional pre-built `manager`**: when given, it's used as-is instead of
     `TaskVineDistributor` constructing its own `vine.Manager(port=port, name=name)` (`port`/`name`
     are then ignored). This is for reusing one manager/port/worker pool across coffea's own
     `dataset_tools.preprocess(fileset, scheduler=...)` (which accepts a dask scheduler - a
     `vine.DaskVine` manager's `.get` method qualifies) and this distributor's own map/reduce
     tasks, rather than standing up two separate managers/worker pools for one script. See
     `examples/ttBar/run_processor_with_vr.py`.
   - **`submit()` gains a `kind` parameter** (`"processor"` or `"reducer"`, see `distributor.py`'s
     `TaskKind`), inserted between `category` and `func`, so `TaskVineDistributor` knows which of
     `resources_processor`/`resources_reducer` to apply without having to infer it from the
     category string's naming convention or from comparing `func` against
     `executor_wrapper`/`reducer_wrapper`. This changes the `Distributor` protocol itself, so
     `LocalDistributor` and `FakeDistributor` (tests/conftest.py) also take and ignore `kind` -
     only `TaskVineDistributor` acts on it. Chosen over inferring from category suffix or from
     `func` identity, both of which would work today but rely on conventions the protocol doesn't
     enforce.
   - Tested against a real `vine_worker` subprocess (external, per the design above), not a fake;
     `tests/test_taskvine_distributor.py` is skipped if `vine_worker` isn't on `PATH`.

8. **Checkpoint db durability**: `CheckpointDB` sets `PRAGMA synchronous = OFF` and batches each
   checkpoint event's insert plus its superseded-row deletes into a single transaction
   (`add_checkpoint`/`delete_checkpoint` take `commit=False`; `Pipeline._checkpoint` commits once
   after all of them). The db is a restart aid, not a durability guarantee: on crash before a
   commit, the worst case is a checkpoint that never got recorded, so its work is redone on
   restart - the same cost as any interval that hasn't been checkpointed yet, not new data loss.
   What the single-transaction batching does protect is atomicity: a restart should never see a
   superseded checkpoint row deleted without its replacement present, or vice versa.

9. **`VineReduce.extra_files`/`environment_variables`**: added so a caller can hand a processor
   its supporting files (e.g. a data file it reads by relative path, an auth token/proxy) and
   env vars (e.g. `X509_USER_PROXY`) without writing distributor-specific code. `VineReduce`
   itself stays distributor-agnostic - it just calls `distributor.add_file(path)`/
   `distributor.set_env_var(name, value)` once per entry, at the very start of `compute()`,
   before any task is submitted. This extends the `Distributor` protocol with those two methods
   (see "API vine_reduce <-> distributor" above):
   - `LocalDistributor.add_file` is a no-op - its subprocesses already share vine_reduce's
     filesystem (see its module docstring), so a local path is already visible under that same
     path without shipping anything; no separate "remote name" exists to place it under.
     `set_env_var` stores into a dict applied via `os.environ.update()` inside the worker
     subprocess itself (`_run_cloudpickled`), not in the parent process, so it takes effect
     regardless of whether the pool already forked that worker by the time `set_env_var` was
     called.
   - `TaskVineDistributor.add_file` calls `manager.declare_file()` once and remembers the
     `(basename, vine.File)` pair; every `submit()` call from then on adds it as a task input
     under that basename, alongside whatever `_remap_files` already attaches. `set_env_var`
     similarly remembers `{name: value}` and calls `task.set_env_var(name, value)` for every task
     submitted after. Only a plain basename is supported (no caller-chosen remote name) - the
     common case (a processor module, a data file, a proxy) is opened by a fixed relative name
     the processor code already expects, and a second knob for renaming isn't needed yet.
