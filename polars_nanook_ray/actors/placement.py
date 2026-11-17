import itertools
import logging
import typing

import ray

from polars_nanook_ray.identity import WorkerId

logger = logging.getLogger(__name__)


class NoRoomForWorkers(RuntimeError):
    pass


def plan_workers(target: int, running: typing.Collection[WorkerId]) -> list[WorkerId]:
    """Return the workers to start for `target` workers to exist.

    `running` holds the existing workers, as `list_workers` returns them, also those
    waiting for room on a node; the new workers take the lowest free indices. Ray
    picks their nodes.
    """
    missing = max(target - len(running), 0)
    free = (w for i in itertools.count() if (w := WorkerId(i)) not in running)
    return list(itertools.islice(free, missing))


def check_room(cpus: float, memory: float) -> None:
    """Raise `NoRoomForWorkers` if a worker's CPUs and memory exceed every alive node's.

    Ray would never place such a worker. A worker that fits a node that is busy waits
    until Ray has room for it.
    """
    nodes = [
        (node["Resources"].get("CPU", 0), node["Resources"].get("memory", 0))
        for node in ray.nodes()
        if node["Alive"]
    ]
    if any(n_cpus >= cpus and n_memory >= memory for n_cpus, n_memory in nodes):
        return
    most_cpus = max((n_cpus for n_cpus, _ in nodes), default=0)
    most_memory = max((n_memory for _, n_memory in nodes), default=0)
    if cpus > most_cpus:
        shortfall = f"{cpus:g} CPU, but the largest node has {most_cpus:g} CPU"
    elif memory > most_memory:
        shortfall = (
            f"{_gib(memory)} of memory, but the largest node has {_gib(most_memory)}"
        )
    else:
        shortfall = (
            f"{cpus:g} CPU and {_gib(memory)} of memory, but no single node has both"
        )
    msg = (
        f"Cannot place a worker on any alive Ray node: it requests {shortfall}. "
        "Lower `cpus_hint` or `memory_hint`, or add larger nodes."
    )
    raise NoRoomForWorkers(msg)


def _gib(memory: float) -> str:
    return f"{memory / 1024**3:g} GiB"


class WorkerPoolReconciler:
    """Keep the worker pool at its target count by starting missing workers.

    Ray restarts a worker actor within its restart limit, on another node if its node
    is lost, but not past that limit. Each `reconcile` starts workers at the lowest
    free indices for the pool to reach the target again.
    """

    def __init__(
        self,
        interval: float,
        start: typing.Callable[[WorkerId], object],
        running: typing.Callable[[], set[WorkerId]],
    ) -> None:
        self.interval = interval
        self._start = start
        self._running = running

    def reconcile(self, target: int) -> list[WorkerId]:
        """Start workers up to `target`; return those started."""
        started = plan_workers(target, self._running())
        for worker in started:
            logger.warning("Replacing worker %s", worker)
            self._start(worker)
        return started
