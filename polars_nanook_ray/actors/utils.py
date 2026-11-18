import logging
import os
import pathlib
import signal
import typing

import ray
from ray.actor import ActorHandle

if typing.TYPE_CHECKING:
    from polars_nanook_ray.identity import WorkerId

logger = logging.getLogger(__name__)

# seconds to wait for an actor call to complete
ACTOR_RESPONSE_TIMEOUT = 30


def _log_to_stderr() -> None:
    """Log this package's records to stderr, which Ray collects, in an actor process.

    Only in our actors: the driver is the user's process, whose logging is theirs to
    set up. A process whose root logger has handlers, e.g. from Ray's
    `logging_config`, keeps its own setup. `LOGLEVEL` sets the level, INFO by default.
    """
    package = logging.getLogger("polars_nanook_ray")
    if logging.getLogger().handlers or package.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    package.addHandler(handler)
    package.setLevel(os.getenv("LOGLEVEL", "INFO").upper())


def _handle_sigterm(func: typing.Callable[[], None]) -> None:
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handler(signum: int, frame: object) -> None:
        try:
            func()
        finally:
            if callable(previous_handler):
                previous_handler(signum, frame)  # type: ignore[arg-type]

    signal.signal(signal.SIGTERM, _handler)


def resolve_actor_handles(
    workers: typing.Collection["WorkerId"], namespace: str
) -> dict["WorkerId", ActorHandle]:
    actors: dict[WorkerId, ActorHandle] = {}

    for worker in workers:
        try:
            actors[worker] = ray.get_actor(worker.actor_name, namespace=namespace)
        except ValueError:
            logger.warning("Actor %s not found, skipping", worker)

    return actors


def terminate_actors(actors: list[ActorHandle], timeout: float) -> None:
    if len(actors):
        try:
            ray.get([actor.stop.remote() for actor in actors], timeout=timeout)
        except Exception:
            logger.exception("One or more actors failed to stop cleanly")
        finally:
            for actor in actors:
                ray.kill(actor)  # no-op if the actor is already gone


def make_dirs(paths: typing.Iterable[str]) -> None:
    """Create the directories `paths` on this node, if missing."""
    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
