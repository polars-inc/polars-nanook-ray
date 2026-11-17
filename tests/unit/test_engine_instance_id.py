from polars_nanook_ray.actors.engine_instance_id import EngineInstanceId
from polars_nanook_ray.actors.scheduler import _resolve_worker_actor_names


def test_each_start_gets_its_own_engine_instance_id() -> None:
    assert EngineInstanceId.new("worker-0") != EngineInstanceId.new("worker-0")


def test_engine_instance_ids_round_trip() -> None:
    instance_id = EngineInstanceId.new("worker-1")

    assert EngineInstanceId.parse(str(instance_id)) == instance_id
    assert str(EngineInstanceId("scheduler")) == "scheduler"


def test_engine_instance_ids_belong_to_their_actor_only() -> None:
    def actor_name(value: object) -> str | None:
        instance_id = EngineInstanceId.parse(value)
        return None if instance_id is None else instance_id.actor_name

    assert actor_name(str(EngineInstanceId.new("worker-1"))) == "worker-1"
    assert actor_name(str(EngineInstanceId.new("worker-10"))) == "worker-10"
    assert actor_name("worker-1") == "worker-1"
    assert actor_name("scheduler") == "scheduler"
    assert actor_name(None) is None


def test_engine_instance_ids_resolve_to_their_worker_actors() -> None:
    workers = {"worker-0", "worker-1", "worker-10"}

    assert _resolve_worker_actor_names(
        {str(EngineInstanceId.new("worker-1")), "worker-10"}, workers
    ) == {"worker-1", "worker-10"}
    assert (
        _resolve_worker_actor_names({str(EngineInstanceId.new("worker-2"))}, workers)
        is None
    )
    assert _resolve_worker_actor_names(None, workers) is None
