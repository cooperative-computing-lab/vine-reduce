"""Live status display for VineReduce.compute(): four status bars per
processor - the processor's own map step, reductions, events, and datasets,
in that order top to bottom - plus one printed line for every finished
processor/reducer task (see Pipeline._report_task / TaskReport in
pipeline.py).

Each bar shows completed (green) / failed (red) / remaining-of-total (gray)
as a single segmented bar, left-justified - no in-flight segment - with the
raw counts right-justified after it, colored completed (yellow) / failed
(red) / total (blue). The events row additionally shows, in parentheses
right after the completed count, how many of those are safe (green) - i.e.
durably checkpointed, see Pipeline.events_safe. The datasets row has no
failed count - it shows completed (green) / total (cyan) instead:

    <processor name> [ ...bar... ]                 completed/failed/total
    reductions       [ ...bar... ]                 completed/failed/total
    events           [ ...bar... ]         completed(safe)/failed/total
    datasets         [ ...bar... ]                          completed/total

`total` for the events bar is exact (the sum of every file's entry count).
`total` for the processor and reductions bars is an estimate, extrapolated
from work done so far - see _proc_tasks_total/_reduce_tasks_total below,
which follow the same approach as dynamic_data_reduction's own
ProcCounts.proc_tasks_total/accum_tasks_total (see
https://github.com/cooperative-computing-lab/dynamic_data_reduction/blob/
6156404b7a2d3da952a7f8159026b468379f4888/src/dynamic_data_reduction/main.py#L501
and #L530): extrapolate the events-per-task ratio seen so far across the
whole dataset for the processor bar, then simulate folding whatever's left
(pending processor tasks plus reductions already produced) `reduction_size`
at a time for the reductions bar. Both are recomputed every refresh, so they
settle down as the run progresses and can occasionally shrink or grow.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .pipeline import OutcomeStatus, ProgressCounters, TaskReport

if TYPE_CHECKING:
    from .pipeline import Pipeline

__all__ = ["NullProgressReporter", "ProgressReporter"]

_BAR_WIDTH = 30
_STATUS_STYLE = {
    OutcomeStatus.SUCCESS: "green",
    OutcomeStatus.RESOURCE_EXHAUSTION: "yellow",
    OutcomeStatus.FAILURE: "red",
}
_MIN_REFRESH_INTERVAL_S = 0.2


def _fmt(value: Any) -> str:
    return str(math.ceil(value)) if isinstance(value, float) else str(value)


def _fmt_resource(measured: Any, allocated: Any) -> str:
    """measured(allocated), e.g. memory_mb=200(500) - or just the measured
    value when this resource has no known allocation (e.g. wall_time_s,
    which isn't a cap a distributor sets)."""
    text = _fmt(measured)
    if allocated is not None:
        text += f"({_fmt(allocated)})"
    return text


class NullProgressReporter:
    """No-op TaskReporter/display, used when VineReduce(progress=False)."""

    def report(self, task: TaskReport) -> None:
        pass

    def refresh(self, pipelines: list["Pipeline"], force: bool = False) -> None:
        pass

    def __enter__(self) -> "NullProgressReporter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


def _segment_widths(counts: list[int], width: int) -> list[int]:
    """Split `width` characters across `counts` proportionally, using
    largest-remainder rounding so the parts always sum to exactly `width`
    (rather than to whatever integer truncation happens to give)."""
    total = sum(counts)
    if total <= 0 or width <= 0:
        return [0] * len(counts)
    raw = [count * width / total for count in counts]
    widths = [int(r) for r in raw]
    remainder = width - sum(widths)
    order = sorted(range(len(counts)), key=lambda i: raw[i] - widths[i], reverse=True)
    for i in order[:remainder]:
        widths[i] += 1
    return widths


def _bar(
    completed: int,
    failed: int,
    total: int,
    width: int = _BAR_WIDTH,
) -> Text:
    """A single-line, multi-color bar: green completed, red failed, and the
    rest of `total` in gray. No in-flight segment - see module docstring."""
    remaining = max(0, total - completed - failed)
    widths = _segment_widths([completed, failed, remaining], width)
    text = Text("[ ")
    for count, chars, style, glyph in zip(
        (completed, failed, remaining),
        widths,
        ("green", "red", "bright_black"),
        ("█", "█", "░"),
    ):
        if chars:
            text.append(glyph * chars, style=style)
    text.append(" ]")
    return text


def _counts_colored(
    completed: int, failed: int | None, total: int, safe: int | None = None
) -> Text:
    """Colored counts text: completed - yellow, or green when `failed` is
    None (the datasets row, which has no failed count) - with, for the
    events row, the safe/checkpointed subset (green) in parentheses right
    after it - then failed (red) if given, then total - blue, or cyan when
    `failed` is None. See module docstring."""
    text = Text()
    text.append(str(completed), style="yellow" if failed is not None else "green")
    if safe is not None:
        text.append("(")
        text.append(str(safe), style="green")
        text.append(")")
    if failed is not None:
        text.append("/")
        text.append(str(failed), style="red")
    text.append("/")
    text.append(str(total), style="blue" if failed is not None else "cyan")
    return text


def _proc_tasks_total(events_total: int, events_submitted: int, submitted: int, failed: int) -> int:
    """Estimated total processor tasks for one processor, extrapolated from
    the events-per-task ratio seen in submissions so far."""
    if events_total == 0:
        return 0
    if events_submitted == 0:
        return 1
    good = submitted - failed
    return math.ceil((events_total / events_submitted) * (good or submitted))


def _reduce_tasks_total(
    proc_total: int,
    proc_completed: int,
    reduce_submitted: int,
    reduce_failed: int,
    reduce_completed: int,
    fold_size: int,
) -> int:
    """Estimated total reduce tasks for one processor: simulate folding
    whatever is still left to produce or combine (pending processor tasks,
    plus reductions already produced but not yet folded further)
    `fold_size` at a time until one final result remains."""
    good = reduce_submitted - reduce_failed
    left = proc_total - proc_completed + good - reduce_completed
    if left <= 0:
        return good
    fold_size = max(2, fold_size)
    total = good
    while left > fold_size:
        folds, left = divmod(left, fold_size)
        total += folds
        left += folds
    if left > 0:
        total += 1
    return total


class ProgressReporter:
    """A TaskReporter that prints one line per finished task (via a rich
    Console) and keeps four live-updating status bars per processor (via a
    rich Live display) refreshed above it. Use as a context manager around
    the scheduling loop; call refresh(pipelines) each cycle and report(task)
    is called by Pipeline itself for every finished task."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._live = Live(console=self._console, refresh_per_second=4, transient=False)
        self._last_refresh = 0.0

    def __enter__(self) -> "ProgressReporter":
        self._live.start(refresh=True)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._live.stop()

    def report(self, task: TaskReport) -> None:
        style = _STATUS_STYLE[task.status]
        allocated = task.resources_allocated or {}
        resources = " ".join(
            f"{key}={_fmt_resource(getattr(task.resources, key), allocated.get(key))}"
            for key in ("cores", "memory_mb", "wall_time_s")
        )
        self._console.print(
            f"[{style}]{task.status:<20}[/{style}] "
            f"{task.kind:<9} {task.processor_name}/{task.dataset_name} {task.description} "
            f"(id={task.result_id[:8]}) "
            f"{resources}"
        )
        if task.status != OutcomeStatus.SUCCESS and task.std_output:
            self._console.print(task.std_output, style="dim", highlight=False)

    def refresh(self, pipelines: list["Pipeline"], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < _MIN_REFRESH_INTERVAL_S:
            return
        self._last_refresh = now
        self._live.update(self._render(pipelines))

    def _render(self, pipelines: list["Pipeline"]) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        table.add_column()
        table.add_column(justify="right")

        by_processor: dict[str, list["Pipeline"]] = {}
        for pipeline in pipelines:
            by_processor.setdefault(pipeline.processor_name, []).append(pipeline)

        for index, (processor_name, procs) in enumerate(by_processor.items()):
            if index:
                table.add_row("", "", "")
            self._add_processor_rows(table, processor_name, procs)
        return table

    def _add_processor_rows(self, table: Table, name: str, procs: list["Pipeline"]) -> None:
        counters = ProgressCounters.summed(procs)
        events_safe = sum(p.events_safe for p in procs)
        events_total = sum(p.events_total for p in procs)

        proc_total = _proc_tasks_total(
            events_total=events_total,
            events_submitted=counters.events_submitted,
            submitted=counters.proc_tasks_submitted,
            failed=counters.proc_tasks_failed,
        )
        proc = (counters.proc_tasks_completed, counters.proc_tasks_failed, proc_total)

        reduce_total = _reduce_tasks_total(
            proc_total=proc_total,
            proc_completed=counters.proc_tasks_completed,
            reduce_submitted=counters.reduce_tasks_submitted,
            reduce_failed=counters.reduce_tasks_failed,
            reduce_completed=counters.reduce_tasks_completed,
            # reduction_size can diverge across a processor's pipelines (each
            # halves independently on resource exhaustion) - the smallest
            # current value gives the most folds, the safer (larger) estimate.
            fold_size=min(p.reduction_size for p in procs),
        )
        reduce_ = (counters.reduce_tasks_completed, counters.reduce_tasks_failed, reduce_total)

        datasets_total = len(procs)
        datasets_completed = sum(1 for p in procs if p.finished)

        table.add_row(name, _bar(*proc), _counts_colored(*proc))
        table.add_row("reductions", _bar(*reduce_), _counts_colored(*reduce_))
        table.add_row(
            "events",
            _bar(counters.events_completed, counters.events_failed, events_total),
            _counts_colored(
                counters.events_completed, counters.events_failed, events_total, safe=events_safe
            ),
        )
        table.add_row(
            "datasets",
            _bar(datasets_completed, 0, datasets_total),
            _counts_colored(datasets_completed, None, datasets_total),
        )
