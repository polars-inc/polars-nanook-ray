"""How the scheduler and the workers are named, on Ray and to the engine."""

import dataclasses
import re
import secrets
import typing

import ray

_WORKER_PREFIX = "worker"

_WORKER_PATTERN = re.compile(
    rf"{_WORKER_PREFIX}-(?P<index>0|[1-9][0-9]*)(?:-[0-9a-f]{{8}})?"
)


@dataclasses.dataclass(frozen=True, order=True)
class WorkerId:
    """A worker's identity, across restarts: its index in the pool.

    Its actor name is `worker-<index>`, and each start of its engine gets an instance
    ID of `<actor name>-<8 hex>`, for a restarted worker to register as a new one. The
    name says nothing of the node the worker runs on, which Ray picks.
    """

    index: int

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            msg = f"Not a worker identity: index {self.index!r}"
            raise ValueError(msg)

    @property
    def actor_name(self) -> str:
        return f"{_WORKER_PREFIX}-{self.index}"

    def new_instance_id(self) -> str:
        """Return the instance ID of a new start of the worker's engine."""
        return f"{self.actor_name}-{secrets.token_hex(4)}"

    def __str__(self) -> str:
        return self.actor_name


@dataclasses.dataclass(frozen=True)
class SchedulerId:
    """The scheduler's identity; there is one, `SCHEDULER`."""

    actor_name: typing.ClassVar[str] = "scheduler"
    # constant across restarts, as the workers name it as their leader
    instance_id: typing.ClassVar[str] = "scheduler"

    def __str__(self) -> str:
        return self.actor_name


SCHEDULER = SchedulerId()

Identity: typing.TypeAlias = WorkerId | SchedulerId


def identify(value: object) -> Identity | None:
    """Return the identity an actor name or engine instance ID names, if any."""
    if not isinstance(value, str):
        return None
    if value in (SCHEDULER.actor_name, SCHEDULER.instance_id):
        return SCHEDULER
    if (m := _WORKER_PATTERN.fullmatch(value)) is not None:
        return WorkerId(int(m["index"]))
    return None


def list_workers(namespace: str) -> set[WorkerId]:
    """Return the live workers in `namespace`, straight from the GCS."""
    return {
        worker
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == namespace
        and isinstance(worker := identify(actor["name"]), WorkerId)
    }
