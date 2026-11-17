from polars_nanook_ray.actors.scheduler import _resolve_worker_actor_names
from polars_nanook_ray.actors.subprocess_runner import (
    engine_instance_id,
    is_engine_instance_of,
)


def test_each_start_gets_its_own_engine_instance_id() -> None:
    assert engine_instance_id("worker-0") != engine_instance_id("worker-0")


def test_engine_instance_ids_belong_to_their_worker_only() -> None:
    assert is_engine_instance_of(engine_instance_id("worker-1"), "worker-1")
    assert is_engine_instance_of("scheduler", "scheduler")
    assert not is_engine_instance_of(engine_instance_id("worker-10"), "worker-1")
    assert not is_engine_instance_of(engine_instance_id("worker-1"), "worker-10")
    assert not is_engine_instance_of(None, "worker-1")


def test_engine_instance_ids_resolve_to_their_worker_actors() -> None:
    workers = {"worker-0", "worker-1", "worker-10"}

    assert _resolve_worker_actor_names(
        {engine_instance_id("worker-1"), "worker-10"}, workers
    ) == {"worker-1", "worker-10"}
    assert (
        _resolve_worker_actor_names({engine_instance_id("worker-2")}, workers) is None
    )
    assert _resolve_worker_actor_names(None, workers) is None
