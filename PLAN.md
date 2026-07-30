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
result_id = submit(priority, category, executor_wrapper | reducer\_wrapper, *args): submit a processor or reduction function call. Category is a string that identifies functions of the same processing/reduction set. Returns a result_id, used later to free_result.
outcome = wait(timeout): wait for a result to be available and return its Outcome (RuntimeFailure | ResourceExhaustion | Success). On timeout return None. outcome.result_id identifies which submit() call this outcome corresponds to.
free_result(result_id): remove resources associated with result_id
chunks_wanted = hungry(): number of additional chunks the distributor could handle given the current resources.
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
results_dir str = "results": Local directory to write results, with one subdirectory per dataset.
results_retrieve bool = True: Whether the distributors should copy results to results_dir.
                              If False, it is assumed that the either the distributor has its own 
                              permanent storage, or that result_postprocess copied the result to an 
                              appropiate location.
distributor Optional[Distributor]: The distributor to use. The default distributor is a local default.
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
