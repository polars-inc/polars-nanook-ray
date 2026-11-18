import itertools
import re
import typing

import ray

WORKER_NAME_PREFIX = "worker"

_WORKER_NAME = re.compile(rf"{WORKER_NAME_PREFIX}-(0|[1-9][0-9]*)")


class NoRoomForWorkers(RuntimeError):
    pass


def worker_name(index: int) -> str:
    """Return the actor name of the worker at `index`; it says nothing of its node."""
    return f"{WORKER_NAME_PREFIX}-{index}"


def list_worker_names(namespace: str) -> set[str]:
    """Return the names of the workers in `namespace`, straight from the GCS.

    Lists those waiting for room on a node too.
    """
    return {
        actor["name"]
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == namespace and _WORKER_NAME.fullmatch(actor["name"])
    }


def by_index(names: typing.Iterable[str]) -> list[str]:
    """Return the worker names given, sorted by their index."""
    return sorted(names, key=lambda name: int(name.rsplit("-", 1)[1]))


def plan_workers(target: int, running: typing.Collection[str]) -> list[str]:
    """Return the names of the workers to start for `target` workers to exist.

    `running` holds the names of the existing workers, as `list_worker_names` returns
    them; the new workers take the lowest free indices. Ray picks their nodes.
    """
    missing = max(target - len(running), 0)
    free = (name for i in itertools.count() if (name := worker_name(i)) not in running)
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
