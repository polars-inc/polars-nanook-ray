"""How the scheduler and the workers are named, on Ray and to the engine."""

import dataclasses
import re
import secrets
import typing

import ray

if typing.TYPE_CHECKING:
    from typing import Self

_WORKER_PREFIX = "worker"
# keeps worker names short; a prefix of a node ID is unique in practice
_NODE_PREFIX_LENGTH = 12

_WORKER_PATTERN = re.compile(
    rf"{_WORKER_PREFIX}-(?P<node>[0-9a-f]{{{_NODE_PREFIX_LENGTH}}})"
    r"-(?P<index>0|[1-9][0-9]*)(?:-[0-9a-f]{8})?"
)


@dataclasses.dataclass(frozen=True, order=True)
class WorkerId:
    """A worker's identity: the node prefix and index of its slot, across restarts.

    Its actor name is `worker-<node>-<index>`, and each start of its engine gets an
    instance ID of `<actor name>-<8 hex>`, for a restarted worker to register as a new
    one. `node` is the first 12 hex digits of the Ray node ID.
    """

    node: str
    index: int

    def __post_init__(self) -> None:
        if (
            type(self.node) is not str
            or type(self.index) is not int
            or _WORKER_PATTERN.fullmatch(self.actor_name) is None
        ):
            msg = f"Not a worker identity: node {self.node!r}, index {self.index}"
            raise ValueError(msg)

    @classmethod
    def of(cls, node_id: str, index: int) -> "Self":
        """Return the identity of the worker at `index` on the Ray node given."""
        return cls(node_id[:_NODE_PREFIX_LENGTH], index)

    @property
    def actor_name(self) -> str:
        return f"{_WORKER_PREFIX}-{self.node}-{self.index}"

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
        return WorkerId(m["node"], int(m["index"]))
    return None


def list_workers(namespace: str) -> set[WorkerId]:
    """Return the live workers in `namespace`, straight from the GCS."""
    return {
        worker
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == namespace
        and isinstance(worker := identify(actor["name"]), WorkerId)
    }
