import dataclasses
import itertools
import logging
import typing

import ray

if typing.TYPE_CHECKING:
    from polars_nanook_ray.config import PolarsRayClusterConfig

WORKER_NAME_PREFIX = "worker"

# keeps worker names short; a prefix of a node ID is unique in practice
NODE_ID_LENGTH = 12

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WorkerSlot:
    """A place for one worker: a Ray node, and its position on that node."""

    node_id: str
    index: int

    @property
    def name(self) -> str:
        return f"{WORKER_NAME_PREFIX}-{self.node_id[:NODE_ID_LENGTH]}-{self.index}"


def _fits(total: float, request: float) -> float:
    return total // request if request > 0 else float("inf")


def _node_capacities(
    nodes: list[dict[str, typing.Any]],
    cpus: float,
    memory: float,
    workers_per_node: int,
) -> dict[str, int]:
    return {
        node["NodeID"]: capacity
        for node in nodes
        if node["Alive"]
        and node["Resources"].get("CPU", 0) > 0
        and (
            capacity := int(
                min(
                    workers_per_node,
                    _fits(node["Resources"]["CPU"], cpus),
                    _fits(node["Resources"].get("memory", 0), memory),
                )
            )
        )
        > 0
    }


def _plan_worker_slots(
    capacities: dict[str, int],
    taken: typing.Collection[str],
    count: int,
) -> list[WorkerSlot]:
    free = (
        slot
        for index in range(max(capacities.values(), default=0))
        for node_id in sorted(capacities)
        if index < capacities[node_id]
        and (slot := WorkerSlot(node_id, index)).name not in taken
    )
    return list(itertools.islice(free, max(count, 0)))


def list_worker_names(namespace: str) -> set[str]:
    """Return the names of the live workers in `namespace`, straight from the GCS."""
    return {
        actor["name"]
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == namespace
        and actor["name"].startswith(f"{WORKER_NAME_PREFIX}-")
    }


def plan_workers(
    config: "PolarsRayClusterConfig",
    target: int,
    running: typing.Collection[str],
) -> list[WorkerSlot]:
    """Return the slots to start workers in for the pool to reach `target` workers.

    `running` holds the names of the live workers, as `list_worker_names` returns
    them. Each slot is on an alive node with CPUs, at most `workers_per_node` per
    node and no more than the node's CPUs and memory fit, and is not taken by a
    running worker. Returns fewer slots when the nodes are full; callers report it.
    """
    missing = target - len(running)
    if missing <= 0:
        return []

    capacities = _node_capacities(
        ray.nodes(),
        config.worker.cpus_hint,
        config.worker.memory_hint,
        config.workers_per_node,
    )
    return _plan_worker_slots(capacities, running, missing)


def warn_no_room(started: int, missing: int) -> None:
    logger.warning(
        "Only room for %d of the %d missing workers; add nodes or raise "
        "workers_per_node",
        started,
        missing,
    )


MAX_REFILL_BACKOFF = 600.0


class WorkerPoolReconciler:
    """Keep the worker pool at its target size by refilling empty slots.

    Ray restarts a worker actor within its restart limit, but not past it, and not
    once its node is gone, as workers are pinned to their node. Each `reconcile`
    starts workers on free slots, a new node's included, for the pool to reach the
    target again. A slot refilled again soon after is backed off exponentially, so
    a worker that keeps failing is not recreated in a tight loop.
    """

    def __init__(
        self,
        interval: float,
        plan: typing.Callable[[int, set[str]], list[WorkerSlot]],
        start: typing.Callable[[WorkerSlot], object],
        running: typing.Callable[[], set[str]],
    ) -> None:
        self.interval = interval
        self._plan = plan
        self._start = start
        self._running = running
        # slot name -> (refills in a row, earliest time to refill it again)
        self._backoff: dict[str, tuple[int, float]] = {}
        self._shortfall = 0

    def reconcile(self, target: int, now: float) -> list[WorkerSlot]:
        """Start workers on free slots up to `target`; return the slots started."""
        running = self._running()
        for name, (_, retry_at) in list(self._backoff.items()):
            if name in running and now >= retry_at:
                del self._backoff[name]  # stayed up through its backoff

        slots = self._plan(target, running)
        shortfall = max(target - len(running) - len(slots), 0)
        if shortfall != self._shortfall and shortfall > 0:
            warn_no_room(len(slots), target - len(running))
        self._shortfall = shortfall

        started = []
        for slot in slots:
            refills, retry_at = self._backoff.get(slot.name, (0, 0.0))
            if now < retry_at:
                continue
            logger.warning("Refilling worker slot %s", slot.name)
            self._start(slot)
            delay = min(self.interval * 2**refills, MAX_REFILL_BACKOFF)
            self._backoff[slot.name] = (refills + 1, now + delay)
            started.append(slot)
        return started
