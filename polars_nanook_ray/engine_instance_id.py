import dataclasses
import re
import secrets
import typing

if typing.TYPE_CHECKING:
    from typing import Self

_START_PATTERN = re.compile(r"^(?P<actor_name>.+)-(?P<start>[0-9a-f]{8})$")


@dataclasses.dataclass(frozen=True)
class EngineInstanceId:
    """The ID the engine knows an actor's binary by: `<actor name>[-<8 hex>]`.

    Workers get a new `start` suffix each time they start, so a restarted worker
    registers as a new instance; the scheduler's ID is its plain actor name.
    """

    actor_name: str
    start: str | None = None

    @classmethod
    def new(cls, actor_name: str) -> "Self":
        """Return the ID of a new start of the actor given."""
        return cls(actor_name, secrets.token_hex(4))

    @classmethod
    def parse(cls, value: object) -> "Self | None":
        """Return the ID in an engine config or scaling request, if it is a string."""
        if not isinstance(value, str):
            return None
        if (m := _START_PATTERN.match(value)) is not None:
            return cls(m["actor_name"], m["start"])
        return cls(value)

    def __str__(self) -> str:
        return (
            self.actor_name if self.start is None else f"{self.actor_name}-{self.start}"
        )
