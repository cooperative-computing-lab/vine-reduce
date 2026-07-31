---
name: taskvine-local-testing
description: >-
  Use when writing, reading, or debugging code in this repo that starts an
  ndcctools.taskvine Manager and needs one or more local workers for testing
  or development (e.g. tests/test_taskvine_distributor.py, or extending
  src/vine_reduce/taskvine_distributor.py). Covers vine.Manager setup, the
  vine.Factory context-manager pattern for local workers, category resource
  limits, and the submit/wait/retrieve loop. Not about production/cluster
  deployment - workers on a real cluster are the caller's responsibility, same
  as this project's own TaskVineDistributor.
---

# TaskVine locally: manager + workers, for testing

TaskVine (`ndcctools.taskvine`, imported here as `vine`) is a
manager/worker distributed-execution system. In this repo it backs
`src/vine_reduce/taskvine_distributor.py`. This skill is about running a
manager and worker(s) **on one machine** for tests and local development -
not about deploying to a real cluster (that boundary is documented in
`taskvine_distributor.py`'s module docstring).

## 1. Start a Manager

```python
import ndcctools.taskvine as vine

manager = vine.Manager(port=0)   # port=0 lets TaskVine pick a free port
manager.enable_monitoring(watchdog=True)  # lets TaskVine kill/report tasks that overrun resources
print(manager.port)  # the port workers need to connect to
```

Use `port=0` for tests so parallel test runs never collide on a fixed port.
See `TaskVineDistributor.__init__` (`src/vine_reduce/taskvine_distributor.py`)
for how this repo wires it up, including the `.port` property.

## 2. Provision local workers with `vine.Factory`

Don't spawn `vine_worker` yourself. `vine.Factory` is the supported way to
provision workers, and it's a context manager: workers start on `with` entry
and are cleaned up automatically on exit, even if the block raises.

```python
workers = vine.Factory(manager=manager)  # batch_type defaults to "local"
workers.cores = 2
workers.min_workers = 1
workers.max_workers = 1

with workers:
    # submit tasks, wait for results - workers are up for the whole block
    ...
# workers are terminated here, on exit from the `with` block
```

- `batch_type="local"` (the default) makes the factory launch plain local
  `vine_worker` processes on this machine - no batch system involved. That's
  the right choice for tests/dev; other `batch_type` values (`"condor"`,
  `"sge"`, ...) are for real clusters and out of scope here.
- Passing `manager=manager` (rather than `manager_host_port=...`) lets the
  factory read the port/name directly off the `vine.Manager` object instead
  of duplicating it.
- `min_workers`/`max_workers` bound how many workers the factory keeps
  alive; for a deterministic test, set both to the same small number.
- The `vine_factory` binary (not `vine_worker` directly) must be on `PATH` -
  it ships with the `ndcctools`/`cctools` pixi dependency. Guard tests that
  need it with `pytest.mark.skipif(shutil.which("vine_factory") is None, ...)`.
- Because results/functions are cloudpickled by reference (see
  `src/vine_reduce/serialization.py`), a worker still needs the right
  `PYTHONPATH` to import whatever modules those functions live in when it
  unpickles them. **Don't** use `Factory`'s `env` option for this (it's a
  literal `--env=VAR=value` passthrough to `vine_factory` and does not reach
  the per-task Python subprocess). Instead, just set `PYTHONPATH` in the
  calling process's own environment before entering the `with` block -
  `Factory.start()` launches `vine_factory` via `subprocess.Popen` with no
  explicit `env=` kwarg, so it inherits this process's environment as-is,
  and that propagates on down through `vine_worker` to the task subprocess.
  In a pytest fixture, use `monkeypatch.setenv("PYTHONPATH", ...)` so it's
  reverted automatically - see `tests/test_taskvine_distributor.py`'s
  `distributor` fixture for the working pattern.

## 3. Category resources

Resource limits (cores/memory/disk) are set **per category**, not per task -
`Manager.set_category_resources_max(category, rmd)`. See
`TaskVineDistributor._configure_category` for the exact mapping this repo
uses from its own resource dict keys (`cores`, `memory_mb`, `disk_mb`) onto
the `rmsummary` field names TaskVine expects (`cores`, `memory`, `disk`).
Call it once per category, the first time a task in that category is
submitted - not on every `task.set_cores(...)`/`task.set_memory(...)` call.

## 4. submit / wait / retrieve loop

```python
task = vine.PythonTask(some_func, *args)
task.set_category("my-category")
manager.set_category_resources_max("my-category", {"cores": 1})  # once per category

taskvine_id = manager.submit(task)

while not manager.empty():
    completed = manager.wait(5)  # seconds; None if nothing finished within timeout
    if completed is None:
        continue
    if completed.successful():
        print(completed.output)
    else:
        print("failed:", completed.result)  # e.g. "resource exhaustion", "max wall time"
```

For the full picture of how `vine_reduce` bridges its own file-token
protocol onto TaskVine's file model (declare_temp/add_output/fetch_file),
read `TaskVineDistributor.submit`/`.wait`/`.retrieve` directly - it's a
complete, working example in this codebase, not just a doc.

## 5. Running this repo's TaskVine tests

```bash
pixi run -e dev pytest tests/test_taskvine_distributor.py -v
```

These tests are skipped automatically (not failed) when the required
TaskVine binary isn't on `PATH`. If you add new tests that provision workers
via `vine.Factory`, guard them the same way, checking for `vine_factory`
rather than `vine_worker`.

## Gotchas

- `manager.wait(timeout)` only accepts an integer (or the string
  `"wait_forever"`) - round a float timeout up (`math.ceil`) rather than
  truncating, or a small positive timeout silently becomes "return
  immediately."
- Set `workers.cores`/`min_workers`/`max_workers` *before* entering the
  `with` block for the initial launch; changing them while the factory is
  already running (inside the block) is supported for the writable subset of
  options (see `Factory._config_file_options`), but the simple case is to
  configure once, up front.
- Don't confuse `Task` and `PythonTask` - `PythonTask` is what wraps a plain
  Python callable (via cloudpickle) for remote execution; that's what this
  repo submits.
