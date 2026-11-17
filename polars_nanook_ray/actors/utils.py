import logging
import pathlib
import secrets
import shutil
import signal
import tempfile
import typing

import ray
from ray.actor import ActorHandle

if typing.TYPE_CHECKING:
    from polars_nanook_ray.identity import Identity, WorkerId

logger = logging.getLogger(__name__)


def _resolve_config_path(identity: "Identity") -> pathlib.Path:
    return pathlib.Path(
        tempfile.gettempdir(),
        f"{identity.actor_name}-{secrets.token_hex(8)}",
        "config.toml",
    )


def _write_config(config_path: pathlib.Path, config: str) -> None:
    config_path.parent.mkdir(mode=0o700)
    config_path.write_text(config)


def _remove_config(config_path: pathlib.Path) -> None:
    shutil.rmtree(config_path.parent, ignore_errors=True)


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
