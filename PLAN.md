# vine_reduce

This file describes the design of the vine_reduce python module, which generates MapReduce-like
workflows for High Energy Physics (HEP). vine_reduce does not itself execute the workflows: it
relies on a **distributor** that manages a distributed high-throughput computation at scale, and on
an **executor** that runs individual functions on the remote worker nodes. See the README for
installation and a runnable quick start; this file is the design reference.

## HEP Workflows

Typical HEP workflows consist of orthogonal processing functions applied to collision events.
Processing functions and collision events are naturally parallel: one processing function does not
affect another, nor does the processing of one event affect another. Since processing a single
event is very fast, events are grouped into sets called chunks, and processing functions are
applied to chunks. The data is organized into datasets. A dataset consists of a name, metadata,
and a set of URLs. The URLs identify files that contain the events. Events in a file are numbered
from [0, num_entries), from which chunks are formed. Chunks never cross file boundaries.

## Reduction and Final Results

**Rule**

- Pool: one per (processor, dataset). A file's chunk outputs join the pool only once *all* of that
  file's chunks have succeeded - never reduced file-by-file first (invariant 1).
- Fold: whenever the pool holds `>= reduction_size` items, the oldest `reduction_size` are reduced
  together in one call. The pool always drains: once chunk generation is exhausted, whatever
  remains - even a single item - is reduced as a smaller final group.
- `is_result(num_events, total_time, total_memory)` is called on the group's totals *before* the
  reduction is submitted. `True` -> this fold is final: `result_postprocess` is applied to its
  output, the output stops being eligible for further reduction, and a new group starts forming.
  Default `is_result`: `True` once the group covers every event of the dataset (one final result
  per dataset).
- Deadlock guard: if a drained single-item group is still rejected by `is_result`, `Pipeline`
  raises `VineReduceError` instead of resubmitting - nothing can ever arrive to change the answer.
  Only reachable with a custom `is_result` whose threshold the dataset's remaining events can
  never reach; the default can't hit it.
- `reduction_size`: per (processor, dataset), default 10, must resolve to an int >= 2 (else
  `VineReduceError` at config time). Halves (floor 2) on distributor resource exhaustion; the
  group is requeued at the front of the pool and retried at the smaller size.
- Reducer contract: associative, distributive, commutative, same output type as the processor.
  Default `f(a, b): a += b; return a` (`default_reducer`, `src/vine_reduce/defaults.py`); the
  wrapper (`reducer_wrapper`, same file) frees each input immediately after folding it in.
- Chunks of different datasets are never reduced together.

**Why**

Reduction is organized around a pool rather than a per-file tree so that a slow or oddly-sized
file doesn't block progress: outputs join a shared pool the moment they're ready, and any
`reduction_size` of them can be folded together regardless of which file they came from. The pool
"always drains" property matters for small or oddly-shaped datasets - a single-chunk dataset must
still produce a result, not wait forever for siblings that don't exist.

`is_result` is deliberately checked on group totals *before* submission, not on the reduction's
own output after the fact: the group's inputs already carry everything needed to decide (see
"Data Flow"), and checking early is what lets a distributor that distinguishes durable from
disposable storage know whether to declare this call's result as a checkpoint at submit time (see
"When a checkpoint is taken").

From the distributor's perspective there is a distinction between a workflow result and a function
outcome. A workflow result is the data the user wants. A function outcome (the `Outcome` union -
see "Dataclasses") is what vine_reduce actually reacts to; it is measured by
`executor_wrapper`/`reducer_wrapper` using core python modules where possible (e.g.
`resource.getrusage`, `time.monotonic`). Workflow results are never read into memory by
vine_reduce, only remotely by the distributor, because they may be too large in memory or
deserialization time.

## Temporary Results, Checkpoints, and Restart

Any intermediate result - a chunk's output, or a reduction's output that is not a final result -
is temporary. Checkpoints make selected intermediates durable so an interrupted run can restart
from them instead of from scratch. vine_reduce does not generate a checkpoint's bytes itself; it
decides *when* to checkpoint and tells the distributor to make that result durable
(`submit(..., is_checkpoint=True)`). A final result is a special kind of checkpoint whose events
are never reduced further.

### Invariants

Three rules govern pooling, durability, and release. The rest of this section, and the pipeline
implementation, refer back to them by number.

**Rule**

1. **Only completely processed files join the pool.** A file's chunk outputs are staged aside and
   enter the reduction pool only once every chunk of that file has succeeded.
2. **A checkpoint's inputs can be freed once it is durable.** When a reduction's output is
   checkpointed (or becomes a final result), everything folded into it is no longer needed for
   recovery and is released. A non-final checkpoint's *own* copy is not released then: it stays
   live as a possible input to later reductions, and is released only once a further checkpoint
   covers it.
3. **An uncheckpointed accumulation's inputs are freed only when something downstream is
   checkpointed** - recursively through its lineage, stopping at prior checkpoints (whose own
   lineages were already freed when they were checkpointed, per invariant 2).

**Mechanics** (all in `src/vine_reduce/pipeline.py`)

- Invariant 1: `_handle_chunk_outcome` stages chunk results in a per-file
  `_files_in_progress[url].staged_items` list and moves them into the pool only when the covered
  events reach the file's `num_entries`. On restart, the invariant holds because a checkpoint row
  only ever covers files whose chunks all succeeded (rows are only written for pool items, which
  invariant 1 already gates), so the restart skip set can never skip a partially-processed file.
- Invariant 2: `_checkpoint` releases a checkpoint's inputs only after durability is established
  (final: `retrieve()` into `results_dir` returned; non-final: the distributor made the copy
  durable at submit time per the `is_checkpoint=True` contract) and after the checkpoint store's
  atomic `record(...)` committed.
- Invariant 3: a reduction's output keeps its whole input group alive as `PoolItem.inputs`; the
  *only* place anything is released is `_checkpoint`, via `_release_covered` - a walk that frees
  an item and descends into its `inputs` exactly while the item is not itself checkpointed. The
  walk uses an explicit stack, not recursion, so an arbitrarily long uncheckpointed chain cannot
  hit Python's recursion limit, and it clears `handle`/`inputs` as it goes, making double-release
  structurally impossible.

### When a checkpoint is taken

**Rule**

A non-final reduction is checkpointed when any of these triggers fires (each is independent; a
threshold left as None disables that trigger):

- `checkpoint_time` (seconds): the group's summed since-last-checkpoint wall time reaches the
  threshold. Wall time is tracked per reduction lineage - each pooled item carries the wall time
  accumulated since its own lineage was last checkpointed - not globally per (processor, dataset).
- `checkpoint_distance` (accumulations): some ancestor in the group has gone at least this many
  reduction folds without being checkpointed. Distance is tracked per pooled item: 0 for a fresh
  chunk result, `max(group) + 1` whenever a group is folded; the trigger uses the *max* across the
  group, not a sum.
- `checkpoint_accumulations=True`: checkpoint every non-final reduction outright, regardless of
  the thresholds.

**Why**

The decision is made *before* the reduction is submitted, not after it returns: at submission
time the group's inputs already carry their since-checkpoint time (summed across the group) and
distance (maxed across the group), and those - together with `is_result` - are enough to decide.
The reduction call's own cost is therefore not part of the go/no-go decision (it is folded into
the resulting item's counters once the call returns); in practice this shifts a threshold
crossing by at most one reduction group. Deciding at submit time is required by the `Distributor`
protocol: `submit(..., is_checkpoint=...)` is when a distributor that distinguishes durable from
disposable storage (e.g. TaskVineDistributor's `declare_file(cache=True)` vs `declare_temp()`)
declares the result's file.

### Where a checkpoint lives

**Rule**

A checkpoint's durability is unconditional - there is no per-checkpoint or per-distributor opt
out. What differs is *where* the durable copy lives and how vine_reduce learns its path:

- A **final result** gets vine_reduce's own naming and location: `_checkpoint` calls
  `retrieve(result_id, dest_path)` with `dest_path` under
  `results_dir/<dataset>/<processor>/<processor>__<uuid4>.pkl.zst`, then immediately releases the
  distributor's copy - nothing ever reduces a final result further, so `results_dir` is the only
  durable copy needed from that point on.
- A **non-final checkpoint**'s durable copy is entirely the distributor's own concern (e.g. each
  shipped distributor's `checkpoint_dir` constructor argument). vine_reduce only learns where it
  ended up, via `checkpoint_path(result_id)`, to record in the checkpoint store. The
  distributor's copy stays live and reusable as a reduction input (so the manager never re-sends
  a checkpoint it already generated), per invariant 2.

**Why**

Durability is unconditional (no per-checkpoint or per-distributor opt-out) because an undurable
checkpoint would be silently unrecoverable after a crash - there would be no way to tell, from the
checkpoint store alone, that the thing it points at was never actually made durable.

### The checkpoint store

Checkpoints are recorded in a sqlite database - the **checkpoint store**
(`CheckpointStore`, `src/vine_reduce/checkpoint_store.py`; the file defaults to
`results_dir/vine_reduce.db`).

**Rule** - schema, versioned via `PRAGMA user_version = 1`:

```sql
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY,
    processor   TEXT    NOT NULL,
    dataset     TEXT    NOT NULL,
    num_events  INTEGER NOT NULL,
    wall_time_s REAL    NOT NULL,
    memory_mb   REAL    NOT NULL,
    is_final    INTEGER NOT NULL,
    path        TEXT    NOT NULL
);
CREATE INDEX checkpoints_by_pair ON checkpoints(processor, dataset);

CREATE TABLE checkpoint_files (
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    file_url      TEXT    NOT NULL,
    PRIMARY KEY (checkpoint_id, file_url)
);

CREATE TABLE dataset_checksums (
    dataset  TEXT PRIMARY KEY,
    checksum TEXT NOT NULL
);
```

Rows are read back by column name (`sqlite3.Row`) into a frozen `CheckpointRecord` dataclass
(`id`, `processor`, `dataset`, `covers_files: frozenset[str]`, `num_events`, `wall_time_s`,
`memory_mb`, `is_final`, `path`), with `covers_files` coming from the `checkpoint_files` join
table - normalized rather than a JSON blob, so it is schema-enforced and cascade-deleted.

Store semantics:

- `record(..., supersedes=[row_ids])` is the **only write path** for checkpoints: it inserts the
  new row (and its `checkpoint_files`) and deletes the rows it supersedes in ONE transaction
  (`with conn:`). There is no `commit=` parameter and no public `commit()` - atomicity is a
  property of the API, not caller discipline. A restart therefore never sees a superseded row
  deleted without its replacement present, or vice versa.
- `dataset_changed(dataset, checksum)`: a checksum of each dataset's definition (name, metadata,
  files) is stored alongside its checkpoints. If it changes between runs, that dataset's
  checkpoint rows are discarded and it restarts from scratch - there is no way to know existing
  checkpoints still correspond to the new definition.
- `PRAGMA synchronous = OFF`: the store trades durability for write throughput. A crash before a
  checkpoint's write commits just means that checkpoint never happened and its work is redone on
  restart - the same cost as any not-yet-checkpointed interval, not new data loss.
- `PRAGMA foreign_keys = ON`, so deleting a superseded checkpoint cascades its
  `checkpoint_files` rows.
- On open, if `PRAGMA user_version` does not match the current schema version, any existing
  tables are dropped and recreated (a `warning` is logged if checkpoints were discarded) - the
  same "discard, don't reconcile" trade `dataset_changed` makes for one dataset, extended to the
  whole store when the schema itself changed.

### Restart

**Rule** (`plan_restart(rows, dataset_files)`, a pure function stating the rules once)

1. Every final row is replayed as a final result.
2. The run is finished iff the final rows cover every dataset file; partial rows are then moot
   and ignored.
3. Otherwise every partial (non-final) row is replayed as a pool item.
4. A file is skipped by chunk generation iff a replayed row covers it.

**Mechanics**

`_seed_from_checkpoints` then executes the plan. Every replayed **partial** row is handed to the
distributor via `adopt_checkpoint(result_id, row.path)` - `result_id` freshly minted by
vine_reduce itself, same as for `submit()` - which registers the on-disk file as if it were a
completed Success result of this run submitted with `is_checkpoint=True` and returns its file
handle, which vine_reduce wraps into a `ResultHandle`. From then on a restart-seeded pool item is
indistinguishable from one this run produced itself: the same handle flows into later `submit()`
args, and the same
`release_result` releases it - one release channel and one file-cleanup owner (the distributor)
cover both cases. A replayed **final** row gets no handle: a final result is never resubmitted,
so it needs no distributor identity. Seeded items start with zeroed since-checkpoint counters -
the row *is* a checkpoint, so nothing has accumulated on top of it yet.

### Releasing results

**Rule**

A distributor-side result is only ever released once a checkpoint covers it - never merely
because it was folded into a later reduction (invariants 2 and 3). Concretely, when
`_checkpoint` runs for a new item:

1. Locate/create the durable copy (final: retrieve into `results_dir`, release the distributor's
   copy, drop the handle; non-final: look up `checkpoint_path`, keep the handle live).
2. `record(...)` the new row, atomically deleting the rows of any inputs that were themselves
   checkpoints (they are superseded).
3. Call `_release_covered` on every input: free it and its whole not-yet-checkpointed lineage,
   stopping at items that are themselves checkpoints. Then clear the new item's `inputs` and
   reset its since-checkpoint counters.

`release_result` is the *only* release channel, and its contract (a hard protocol requirement,
not an implementation detail) is that it also removes a checkpoint's durable on-disk copy - for a
this-run checkpoint or an adopted one alike. The distributor is the single owner of non-final
checkpoint files; `Pipeline` never deletes one itself. Final results in `results_dir` are
vine_reduce's own and are never deleted.

**Why**

Ordering matters: the store stops pointing at superseded rows (step 2 commits) *before* their
files are removed (step 3), so a crash in between can only leave a not-yet-deleted file the
store no longer references - which restart tolerates - never a referenced-but-deleted file.

### Pipeline state

The pipeline's unit of bookkeeping is `PoolItem` (`src/vine_reduce/pipeline.py`), whose fields
each have exactly one meaning:

```python
PoolItem:
handle Optional[ResultHandle]: the item's live distributor identity - handle.file goes into a
                               later submit()'s args, handle.result_id into release_result/
                               retrieve/checkpoint_path. None only once nothing will ever need
                               the distributor's copy again (a final result safely in
                               results_dir, a released item, or a restart-seeded final row).
num_events / wall_time_s / memory_mb: totals accumulated into this item.
files frozenset[str]: dataset file URLs whose data this item represents.
since_checkpoint_time float: wall time accumulated since this lineage was last checkpointed.
since_checkpoint_distance int: reduction folds since this lineage was last checkpointed.
checkpoint Optional[CheckpointRef]: the durable identity (store row_id + on-disk path), present
                               iff the item is durable.
inputs List[PoolItem]: the items folded together to produce this one (empty for a chunk's
                               output). Kept so a lost, not-yet-durable item can be recovered by
                               re-folding its inputs instead of recomputing from scratch;
                               cleared once a checkpoint covers it (invariant 3).
```

## Priorities

**Rule**

All processing calls of the same (processor, dataset) share one priority (larger integer runs
first). A processor declared earlier gets better priority than a later one. Reductions work the
same way, but always outrank every processing call, at any processor's priority level.

**Why**

The goal is to finish one (processor, dataset) before moving to the next, while still overlapping
long tails when resources are available - and reductions outrank processing so that durable
results (and, eventually, freed memory) land as soon as a pool has enough to fold, rather than
queuing behind a backlog of fresh chunk work.

## Chunksize

Chunksize is managed per (processor, dataset). An initial chunksize can be given globally, per
processor, or per dataset; when more than one applies, the most specific wins (per-dataset over
per-processor over the global default). If none is given, all events of a file form one chunk.
Chunksize is dynamic: when the distributor reports that a processing call exhausted its
resources, the chunksize is halved (down to a minimum of 1) and the failed chunk is retried -
re-split first if it predates the halving, so the retry actually runs at the smaller size.

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
│ checkpoint logic (checkpoint_time / checkpoint_distance thresholds)          │
│   ├──► checkpoint store   (sqlite: progress, checksums, restart state)       │
│   ├──► distributor.release_result(result_id)  (superseded temp results)      │
│   └──► results_dir/ (distributor.retrieve() copies a final result here) or   │
│         wherever the distributor durably stores a non-final checkpoint       │
│         (submit(..., is_checkpoint=True) - distributor.checkpoint_path()     │
│         reports where)                                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                             │
                             │  submit(result_id, priority, category, kind,
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
│ executor.submit(processor, args, dataset_metadata=, distributor_metadata=,   │
│                  executor_metadata=).result()                                │
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

The pieces, in flow order (each is user-overridable unless noted):

- **input description** (data): an arbitrary text file (or already-parsed dict) describing
  datasets, files, and metadata per dataset. User given.
- **input_to_datasets** (local function): converts the input description into
  `{dataset_name: {"metadata": {...}, "files": {url: num_entries}}}`. The default accepts that
  dict directly, or loads one from a json file path.
- **datasets** (data, generated by vine_reduce): the dict above. Its per-dataset checksum in the
  checkpoint store controls restart validity (see "The checkpoint store").
- **datasets_to_chunks** (local generator): yields chunks one by one, restarted per processing
  function. Chunks are not all generated at once, so the chunksize can adapt. Generation is
  throttled by `distributor.capacity()` (how many chunks the distributor can take right now),
  capped by `max_chunks_active` minus chunks currently in flight, and by `max_chunks_cycle` per
  scheduling cycle. The default generates chunks at the current chunksize for the
  (processor, dataset).
- **chunk_to_args** (remote function): converts a `Chunk(url, start, stop)` into the data the
  processor is applied to. Takes a mandatory `dataset_metadata` argument and an optional
  `distributor_metadata` argument, which may carry `{"cores": ..., ...}` - `Pipeline` fills it in
  per submission from `distributor.resources("processor")` (see "API vine_reduce <-> distributor"),
  a *static* cap the distributor can report ahead of dispatch (e.g. TaskVineDistributor's
  configured category limit). This is a default only: the execution site may report a more
  precise, real allocation at dispatch time (e.g. TaskVine's worker setting the `CORES`
  environment variable to whatever its own scheduling algorithm actually handed this task, which
  can be less than the configured cap) - see `DaskExecutor`'s `_num_workers` in "Executors and
  remote environments", which prefers that over `distributor_metadata`.
- **executor** (remote object): an `Executor` protocol instance (`submit`/`map`/`shutdown`, named
  after `concurrent.futures.Executor`) that calls the processor on `chunk_to_args`' output.
  `executor_wrapper` calls `executor.submit(processor, args, dataset_metadata=...,
  distributor_metadata=..., executor_metadata=...).result()`, then shuts the (freshly
  deserialized, per-call) instance down. Default: `SimpleExecutor()` - see "Executors and remote
  environments".
- **executor_wrapper** (remote, generated by vine_reduce, not user-overridable): calls
  `chunk_to_args` and `executor`, measures resources, traps any exception (capturing the
  traceback), and serializes the result to a file given as its first argument - a path chosen
  and maintained entirely by the distributor. Serialization is cloudpickle + zstd, streamed
  rather than buffered (`src/vine_reduce/serialization.py`), so results - and the functions that
  produce them - may be closures or lambdas. The `Outcome` carries the path/handle to that
  file on `Success`, never the result itself, so vine_reduce never deserializes workflow
  results. The file is written even on failure, so a distributor that declares it as a required
  task output (TaskVine) does not mask the wrapper's own outcome with a missing-output error.
- **is_result** (local function): decides whether the group about to be reduced becomes a final
  result. See "Reduction and Final Results".
- **result_postprocess** (remote function, optional): applied to a reduction's output when
  `is_result` returned True for its group.
- **reducer_wrapper** (remote, generated by vine_reduce, not user-overridable): folds its input
  files together with the reducer, applies `result_postprocess` for a final result, and - like
  `executor_wrapper` - measures, traps, and serializes to the distributor-chosen file.

## Progress Reporting

**Rule**

- `VineReduce(progress=True)` (the default) shows a live rich-based display while `compute()`
  runs: four status bars per processor - events, the processor's own map step, reductions, and
  datasets - plus one printed line per finished processor/reducer task, with its resource usage
  and success/failure status (plus its captured stdout, on failure). `progress=False` swaps in a
  no-op reporter, for a quiet run (e.g. under a test harness or a non-interactive batch log).
- Two pieces, both introduced by `src/vine_reduce/progress.py`: `ProgressReporter` (a rich
  `Console` for the per-task print line, plus a rich `Live` display for the bars, refreshed at
  most every 0.2s) and `NullProgressReporter` (the same interface - `report`, `refresh`, used as a
  context manager - doing nothing; what `progress=False` uses instead).
- `TaskReporter` (a `Protocol`) / `TaskReport` (a frozen dataclass) are defined in `pipeline.py`,
  not `progress.py`, since `Pipeline` is what needs the interface: `Pipeline._report_task` calls
  `report(TaskReport(...))` once per finished processor/reducer task, right when its `Outcome` is
  examined - before the pipeline acts on it (pooling a chunk's output, folding a reduction,
  retrying, or raising).
- The bars read `Pipeline`'s state through 13 read-only counter properties
  (`events_completed`/`failed`/`submitted`/`safe`, `proc_tasks_completed`/`failed`/`submitted`/
  `in_flight`, `reduce_tasks_completed`/`failed`/`submitted`/`in_flight`) - raw facts only;
  `progress.py` owns all bar-rendering and totals-estimation math (`_proc_tasks_total`/
  `_reduce_tasks_total`, which extrapolate an estimated total from work done so far, the same
  approach as dynamic_data_reduction's own `ProcCounts`).
- `engine.py`'s `_run` calls `reporter.refresh(pipelines)` once per scheduling cycle (and once
  more, forced, right before returning), summing every pipeline sharing a `processor_name` into
  that processor's four bars.

**Why**

`TaskReporter` is defined in `pipeline.py` rather than `progress.py` because `Pipeline` is the
consumer of the interface, not the producer - the protocol lives next to what needs it, and
`progress.py` supplies an implementation, so `Pipeline` never imports `progress.py` at all.

The 13-property surface is a deliberate, narrow coupling, not an event bus or a reach into private
state: nothing in the scheduling loop itself (`engine.py`'s `_run`) reads them, only
`progress.py` does. The cost is informal rather than enforced: adding a new counter to `Pipeline`
means remembering to thread it through `progress.py`'s estimators by hand. This was reviewed
2026-09-03 and judged an acceptable trade-off - the alternative (an event bus, or `progress.py`
reaching into private state) would be more machinery for the same information, and the existing
comment in `pipeline.py` already states the contract ("raw facts only").

## API vine_reduce <-> distributor

A distributor implements the `Distributor` protocol (`src/vine_reduce/distributor.py`):

```python
submit(result_id, priority, category, kind, executor_wrapper | reducer_wrapper, *args,
    is_checkpoint=False): submit a processor or reduction call, identified by result_id - a
    caller-minted id (vine_reduce mints uuid4().hex per call; see "Pipeline state"), unique for
    the lifetime of this distributor. The distributor itself never computes or hands back an id -
    it just remembers result_id for release_result/retrieve/checkpoint_path, and echoes it back
    on the matching Outcome.result_id from wait(). category is a string grouping calls of the
    same processing/reduction set (e.g. for logging or scheduling heuristics). kind is
    "processor" or "reducer", letting a distributor apply different resource requests to each.
    is_checkpoint marks a call whose result must be durable - decided at submit time because that
    is when a distributor declares the result's file (see "When a checkpoint is taken"). func is
    always called as func(dest_file, *args): the distributor picks dest_file (or an opaque token
    standing in for it) and prepends it itself - vine_reduce never picks the path, it only sees
    it echoed back on Outcome.file.
outcome = wait(timeout): block until a submitted call finishes and return its Outcome
    (Success | RuntimeFailure | ResourceExhaustion); None on timeout. outcome.result_id
    identifies which submit() call it corresponds to.
release_result(result_id): release any resources held for result_id. A hard requirement of the
    protocol, not just cleanup: for a result submitted with is_checkpoint=True, or adopted via
    adopt_checkpoint, release_result must also remove the checkpoint's durable on-disk copy -
    the distributor is the single owner of that file. Both shipped distributors behave this way;
    vine_reduce relies on it rather than ever deleting a checkpoint file itself.
file = adopt_checkpoint(result_id, path): register path - an existing durable checkpoint file
    written by a previous run and recorded in the checkpoint store - under result_id, as if it
    were a completed Success result of this run submitted with is_checkpoint=True. result_id
    becomes valid for release_result/retrieve/checkpoint_path exactly like one from submit()
    (checkpoint_path returns path). Returns the distributor's own file handle - the same kind of
    value as Outcome.file - which vine_reduce wraps into a ResultHandle(result_id, file) for use
    inside a later submit()'s args. This makes a restart-seeded pool item indistinguishable from
    one this run produced: one release channel and one file-cleanup owner cover both cases, with
    no distributor-side guessing about which task arguments are restart-seeded on-disk paths.
metadata = resources(kind): a default resource dict (e.g. {"cores": ...}) for calls of this kind,
    or None if this distributor has no meaningful default. A static cap known ahead of dispatch
    (e.g. TaskVineDistributor's configured category limit via resources_processor/
    resources_reducer; LocalDistributor always reports {"cores": 1}, since it runs on the same
    machine as vine_reduce itself, often a shared frontend). Pipeline calls this once per
    processor submission and passes the result as chunk_to_args/executor_wrapper's
    distributor_metadata argument - see "Data Flow". It is a fallback default only: the execution
    site itself may report a more precise, real allocation at dispatch time (e.g. TaskVine's
    worker setting the CORES environment variable), which DaskExecutor's _num_workers prefers
    over this value.
chunks_wanted = capacity(): number of additional chunks the distributor could usefully accept
    right now.
retrieve(result_id, dest_path): copy/materialize the file for a completed (Success) result_id to
    dest_path, a path local to the vine_reduce process. Used for final results, whose location
    (results_dir) and naming are vine_reduce's own convention - vine_reduce always calls this
    rather than assuming Outcome.file is directly readable, keeping the interface correct for a
    distributor that doesn't share a filesystem with vine_reduce.
path = checkpoint_path(result_id): local, durable on-disk path for a completed (Success)
    result_id submitted with is_checkpoint=True (or adopted). Unlike retrieve(), the distributor
    chooses this path itself; vine_reduce calls this for a non-final checkpoint to learn where
    it ended up, to record in the checkpoint store.
add_file(local_path, remote_path=None): make local_path available, under remote_path (default:
    local_path's basename), wherever every call submitted from now on runs. A no-op for a
    distributor whose workers already share vine_reduce's filesystem.
set_env_var(name, value): set an environment variable for every call submitted from now on.
shutdown(): release whatever resources this distributor owns (worker pools, temp directories,
    ...). Also reachable via `with distributor: ...`.
```

`add_file`/`set_env_var` are called once per entry in `VineReduce.extra_files` /
`environment_variables`, at the very start of `compute()`, before any task is submitted - so a
caller can hand a processor its supporting files (e.g. a data file read by relative path, an
auth token/proxy) and env vars (e.g. `X509_USER_PROXY`) without VineReduce knowing anything
distributor-specific.

## Dataclasses

`VineReduce` lives in `src/vine_reduce/engine.py`, alongside `compute()` - the orchestration entry
point and scheduling loop (see "Data Flow" below). Everything else in this section (`Chunk`,
`Outcome` and its variants, `ResultHandle`, `RawOutcome`) lives in `src/vine_reduce/types.py`.

```python
VineReduce:
processors Dict[str, Callable]: Mapping from processor names to processing functions.
input str | dict: pathname to the input description, or an already-parsed dict of that shape.
input_to_datasets Optional[Callable]: Convert input into the dictionary of datasets.
datasets_to_chunks Optional[Callable]: Generate chunks per dataset. Reset per processor.
chunk_to_args Callable: Instantiate chunks.
executor Executor = SimpleExecutor(): Call processor on instantiated chunks. See "Executors" below.
reducer Callable = default_reducer: Function to merge two results together.
reduction_size int | dict = 10: Results to reduce together in a single reduction call. Either a
                               plain int, or {"default": int, "processors": {name: int},
                               "datasets": {name: int}} for per-processor/per-dataset overrides
                               (most specific wins). Must resolve to an int >= 2.
is_result Optional[Callable] = None: is_result(num_events, total_time, total_memory) decides
                               whether the output of a reduction call is a final result or keeps
                               being reduced. Default: True only once all events of the dataset
                               are consumed.
result_postprocess Optional[Callable] = None: Applied to final results.
checkpoint_time Optional[float]: Wall time (seconds), per reduction lineage, that triggers a
                               checkpoint.
checkpoint_distance Optional[int]: Accumulations (reduction folds) since a lineage was last
                               checkpointed that trigger a checkpoint - fires once some ancestor
                               in the group being folded has gone this many uncheckpointed.
checkpoint_accumulations bool = False: If True, checkpoint every non-final reduction result,
                               regardless of checkpoint_time/checkpoint_distance.
results_dir str = "results": Local directory for final results, one subdirectory per dataset
                               and, within that, one per processor (so multiple processors over
                               the same dataset don't collide). Non-final checkpoints are not
                               written here - their durable copy is the distributor's own
                               concern (see "Where a checkpoint lives").
distributor Optional[Distributor]: The distributor to use. Defaults to a LocalDistributor that
                               compute() creates and tears down itself (with checkpoint_dir
                               defaulted to results_dir/checkpoints, next to the checkpoint
                               store, so checkpoints survive to the next run of the same
                               results_dir).
chunksize int | dict | None: Target events per chunk, same dict shape as reduction_size. None
                               means one chunk per file. Halved automatically on resource
                               exhaustion, for chunks not yet generated.
max_chunks_active int = 1000: Cap on chunks in flight (submitted but not yet finished) across
                               all pipelines at once.
max_chunks_cycle int = 100: Cap on new chunks submitted per scheduling cycle, across all
                               pipelines.
db_path Optional[str]: Path to the checkpoint store's sqlite file; defaults to
                               results_dir/vine_reduce.db.
extra_files List[str] = []: Local paths made available, under their basename, to every
                               processor/reducer call, via add_file() at the start of compute().
environment_variables Dict[str, str] = {}: Environment variables set for every processor/reducer
                               call, via set_env_var() at the start of compute().
progress bool = True: Whether to show the live status bars (events, processing, reductions,
                               datasets - four per processor) and print one debug line per
                               finished processor/reducer task, with its resource usage and
                               success/failure status (plus its captured stdout, if any, on
                               failure). See "Progress Reporting" below. Set False for a quiet
                               run, e.g. under a test harness or a non-interactive batch log.
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
  result_id: the id vine_reduce passed to the submit() call this outcome corresponds to.
  resources Dict[str, Any]: resources used by the task, e.g.
                            {"cores": ..., "memory_mb": ..., "wall_time_s": ...}.
                            Measured by executor_wrapper/reducer_wrapper using core python
                            modules where possible (e.g. resource.getrusage, time.monotonic).

RuntimeFailure additionally carries:
  traceback str: captured traceback of the processing/reduction function failure.

Success additionally carries:
  file str: the distributor's own handle for where executor_wrapper/reducer_wrapper serialized
            its result - a real path for a distributor whose workers share vine_reduce's
            filesystem, or an opaque token otherwise (see TaskVineDistributor below). Passed to
            Distributor.retrieve() to materialize it locally.
```

```python
ResultHandle (frozen):
result_id str: for release_result/retrieve/checkpoint_path. Minted by vine_reduce itself
               (uuid4().hex), never by the distributor - see "API vine_reduce <-> distributor".
file str: the distributor's opaque handle (an Outcome.file), for use inside a later submit()'s
          args.
```

Worker-side, `executor_wrapper`/`reducer_wrapper` actually return a `RawOutcome` (`status`,
`resources`, `file | traceback`) - the distributor-agnostic value produced before a `result_id`
is attached. A distributor attaches the `result_id` it was given at `submit()` time via
`RawOutcome.to_outcome(result_id)` to produce the `Outcome` it hands back from `wait()`.

## Distributors

Two `Distributor` implementations ship with vine_reduce.

### LocalDistributor (`src/vine_reduce/local_distributor.py`)

The default when `distributor=` is omitted: runs every processor/reducer call in a local
`ProcessPoolExecutor`, for development, testing, and as the minimal reference implementation of
the protocol.

- Worker "nodes" are local subprocesses sharing vine_reduce's filesystem, so `retrieve()` is a
  plain file copy and `add_file()` is a no-op. Env vars from `set_env_var` are applied inside
  each worker call, so they take effect regardless of when the pool forked its workers.
- Every result lands at a real path the moment it's produced, so `checkpoint_path()` is a
  lookup, no copy. Every result's on-disk filename is its own fresh `uuid4().hex`, independent of
  `result_id` - *which directory* it lands in is what depends on `is_checkpoint`: an ordinary
  result lands under `work_dir` (scratch - a fresh temp directory removed on `shutdown()` unless
  the caller supplied its own), a checkpoint under the constructor's `checkpoint_dir` (default
  `"checkpoints"`, matching TaskVineDistributor), which `shutdown()` never removes. Minting the
  filename itself, rather than deriving it from `result_id`, keeps a checkpoint's name safe from
  collision with a still-live checkpoint from an earlier run regardless of what the caller's
  `result_id`s look like. The directory split is what makes restart possible: a checkpoint must
  still exist, at the path recorded in the checkpoint store, the next time the same
  `results_dir`/`db_path` is used.
- `adopt_checkpoint(result_id, path)` - the reference implementation of the protocol method -
  just records `path` under `result_id` (workers share the filesystem, so `path` is usable
  as-is); the seeded item is then released/retrieved/resubmitted through exactly the same code
  paths as a this-run result.
- `func`/`args` are cloudpickled before crossing into the subprocess, so
  `processor`/`reducer`/etc. may be closures or lambdas, not just module-level callables.
- Priority is best-effort only: a pending call waits in a priority queue until a worker slot is
  free, but once dispatched it cannot be preempted by a higher-priority call submitted later.
- `resources(kind)` always returns `{"cores": 1}`: this runs on the same machine as vine_reduce
  itself, often a shared frontend, so a task must not assume it can use every core in the pool -
  one pool slot is meant to hold one task's work.

### TaskVineDistributor (`src/vine_reduce/taskvine_distributor.py`)

Runs vine_reduce across a real cluster of machines via
[TaskVine](https://cctools.readthedocs.io/en/stable/taskvine), instead of local subprocesses.

- **Manager-only, external workers.** The constructor starts a `vine.Manager` and nothing else;
  `vine_worker` processes, factories, or batch submission are the caller's responsibility, the
  normal way TaskVine is used. A pre-built `manager` (e.g. a `vine.DaskVine`) can be passed in
  instead, so vine_reduce's tasks can share one manager/port/worker pool with a caller's other
  tasks (e.g. coffea's `dataset_tools.preprocess(scheduler=...)`). A manager this class built is
  its own to close: `shutdown()` drops the reference so the listening port is freed right away;
  a caller-supplied manager is left alone.
- **File-passing via opaque tokens.** TaskVine workers generally don't share a filesystem with
  the manager or each other, so a real path can't stand in for `Outcome.file`. Every result
  becomes either a `manager.declare_temp()` file, kept at/near the worker that produced it, or -
  when `submit(..., is_checkpoint=True)` - a `manager.declare_file(path, cache=True)` file with
  `path` a fresh `uuid4().hex` name under the constructor's `checkpoint_dir` (minted independently
  of `result_id`, for the same cross-run collision reason as LocalDistributor). TaskVine transfers
  a `declare_file()` output back to the manager unconditionally as soon as the task completes
  (unlike a temp file, which stays remote until fetched), so a checkpoint is already durably on
  local disk by the time `wait()` reports success - `checkpoint_path(result_id)` is a lookup, no
  copy. Either way, `Outcome.file` is an opaque token this class derives from `result_id` (e.g.
  `"result_<uuid>.p"`), never the real location. When that token later appears in a `submit()`
  call's args (a `reducer_wrapper`
  `input_files` entry), it is recognized by lookup, the underlying `vine.File` is attached as a
  task input under a fresh sandbox name, and that name is substituted into the args actually
  sent - the remote wrapper opens a name that exists in its own sandbox, never the manager-side
  token. `retrieve()` (used for final results) reads bytes binary-safe via
  `manager.fetch_file()` + `File.contents()` - unlike `Manager.fetch_file()`'s own return value,
  which round-trips through a C string and truncates on embedded NUL bytes.
- **Release covers every copy.** `release_result()` undeclares the `vine.File` and, if the
  result was a checkpoint (this-run or adopted), removes its `checkpoint_dir` mirror from local
  disk. It is also called internally for a task that fails or is resource-exhausted (not just a
  `Success`), since that task's declared file/checkpoint bookkeeping would otherwise leak for
  the rest of the run.
- **Adoption, not sniffing.** `adopt_checkpoint(result_id, path)` derives a token from
  `result_id` exactly like `submit()` would, declares `path` under it (`cache=True`), and records
  its checkpoint path -
  so a restart-seeded item flows through remapping/release/retrieve with zero special cases, and
  `_remap_files` only ever matches known tokens, never guesses from path-shaped strings.
- **Infra-level resource exhaustion is mapped, not just Python-level.** TaskVine's resource
  monitor (`enable_monitoring(watchdog=True)`) can kill and report a task that overruns its
  allocation - something a plain `ProcessPoolExecutor` can't detect. `wait()` trusts the
  in-process `RawOutcome` only when `task.successful()`; otherwise it maps TaskVine's own result
  string (`"resource exhaustion"`, `"max wall time"`, `"disk alloc full"` ->
  `ResourceExhaustion`, anything else -> `RuntimeFailure`), so chunksize/reduction_size halving
  is reachable from real cluster failures too.
- **Resources are per-category, not per-task.** `resources_processor`/`resources_reducer` (each
  an optional `{"cores", "memory_mb", "disk_mb"}` dict) are constructor args, applied via
  `manager.set_category_resources_max(category, ...)` the first time each distinct category
  string is seen. `submit()`'s `kind` parameter selects which of the two applies, since the
  caller-supplied category string is not a reliable signal on its own. `resources(kind)` exposes
  the same dict to vine_reduce (via `distributor_metadata`, see "API vine_reduce <-> distributor")
  as a configured cap - TaskVine's own scheduling algorithm, run manager-side at dispatch time,
  decides the real per-task allocation, which can be less; its worker reports that decision via
  the `CORES` environment variable, which `DaskExecutor` prefers over this cap.
- `add_file`/`set_env_var` remember what's been added and attach it
  (`declare_file`/`add_input`, `set_env_var`) to every task submitted from then on. An optional
  poncho `environment` (see "Executors and remote environments") is likewise attached to every
  task.
- Tested against a real `vine_worker` subprocess (skipped if not on `PATH`), not a fake - see
  the `taskvine-local-testing` skill and `tests/test_taskvine_distributor.py`.

## VineReduceCoffea

`VineReduceCoffea` (`src/vine_reduce/coffea.py`) is a `VineReduce` specialization for
[coffea](https://coffeateam.github.io/coffea/)-based workflows over NanoEvents. It only supplies
the coffea-specific pieces; chunking, checkpointing, and restart are inherited unchanged from
`VineReduce`:

- `input_to_datasets` defaults to `coffea_input_to_datasets`, which converts the output of
  coffea's own `preprocess()` (files described by `{"num_entries": ..., "steps": ...,
  "uuid": ...}`) into vine_reduce's `{url: num_entries}` shape. Accepts that dict directly, or a
  path to a json file holding it.
- `chunk_to_args`/`executor` are built in `__post_init__` from
  `schema`/`mode`/`uproot_options`/`object_path` (`chunk_to_args` opens
  `NanoEventsFactory.from_root` over `[chunk.start, chunk.stop)`) and from `processor_args`
  (`executor` is a `CoffeaExecutor(processor_args=...)`, a `SimpleExecutor` subclass that calls
  `processor(events, **processor_args)` then recursively materializes any virtual awkward arrays
  in the result before it is pickled).
- `reducer` defaults to a coffea-flavored `default_reducer` (handles `Addable`, `MutableSet`,
  and recursive `MutableMapping` merging), since base VineReduce's `a += b` default can't merge
  the dict-of-histograms shape coffea processors typically return.

## Executors and remote environments

`executor` (`src/vine_reduce/executor.py`) is a second, smaller axis of pluggability, distinct
from `Distributor`: `Distributor` decides *where* a call runs (which worker); `Executor` decides
*how* the call runs once it's there. It's an `Executor` protocol instance - `submit`/`map`/
`shutdown`, named after `concurrent.futures.Executor` - configured once in the local process and
cloudpickled fresh into every remote call, where `executor_wrapper` uses it as
`with executor: executor.submit(...).result()`.

- **Must always pickle cleanly.** Any live resource an implementation holds (e.g. a process pool)
  is created lazily on first `submit`/`map` and dropped before pickling (see
  `CloudpickleExecutor.__getstate__`), so a configured - even previously-used - instance always
  pickles cleanly.
- **`SimpleExecutor()`** (default): calls `processor(args)` directly, in the same process running
  `executor_wrapper`.
- **`CloudpickleExecutor()`**: isolates the call in its own subprocess (via a
  `CloudpickleProcessPoolExecutor` - cloudpickle rather than stdlib pickle, so `processor` may be
  a closure or lambda), so a crash or memory leak in `processor` doesn't take down the worker task.
- **`DaskExecutor(num_workers=None)`**: for a `processor` that returns a dask-delayed object (or
  array/dataframe); computes it at the execution site via `.compute(num_workers=...)`. dask is not
  a vine_reduce dependency - it must already be installed wherever this executor actually runs. If
  `num_workers` isn't given explicitly, `_num_workers` resolves it in priority order: the `CORES`
  environment variable, when set (the execution site's own report, at dispatch time, of what it
  actually handed this task - e.g. TaskVine's worker echoing its manager's real per-task
  scheduling decision, which can be less than any configured cap); else
  `distributor_metadata["cores"]`, the distributor's static default (see "API vine_reduce <->
  distributor"); else every core on the machine (`os.process_cpu_count()`).

All three inherit `map()` (defined once, in terms of `submit()`) and `__enter__`/`__exit__`
(defined once, in terms of `shutdown()`) from a private `_ExecutorBase`, so each only implements
`submit`/`shutdown` itself. See the README's "Executors" section for the full rundown.

**Remote environments**

Workers need nothing beyond a distributor pre-installed. `get_environment()`
(`src/vine_reduce/remote_environment.py`) packs the calling conda environment (`$CONDA_PREFIX`
by default) into a relocatable [poncho](https://cctools.readthedocs.io/en/stable/poncho) tarball
via `poncho_package_create`, for `TaskVineDistributor(environment=...)` to ship and activate on
every worker. Builds are cached on disk and rebuilt automatically whenever a locally-editable
package (vine_reduce itself, by default, or anything else named via `pip_editable`) has
uncommitted changes, so a tarball never silently ships stale code. See the README's "Packaging
an environment for remote workers".
