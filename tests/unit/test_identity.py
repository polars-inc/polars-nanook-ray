import re

import pytest

from polars_nanook_ray.identity import SCHEDULER, WorkerId, identify


def test_worker_actor_name_is_its_index() -> None:
    assert WorkerId(0).actor_name == "worker-0"
    assert WorkerId(12).actor_name == "worker-12"


def test_worker_instance_id_is_its_actor_name_and_a_start() -> None:
    instance_id = WorkerId(3).new_instance_id()

    assert re.fullmatch(r"worker-3-[0-9a-f]{8}", instance_id)


def test_each_start_gets_its_own_instance_id() -> None:
    worker = WorkerId(0)

    assert worker.new_instance_id() != worker.new_instance_id()


@pytest.mark.parametrize("worker", [WorkerId(i) for i in (0, 1, 10, 123)], ids=str)
def test_worker_names_and_instance_ids_resolve_to_the_worker(worker: WorkerId) -> None:
    assert identify(worker.actor_name) == worker
    assert identify(worker.new_instance_id()) == worker


def test_scheduler_is_named_scheduler_on_ray_and_to_the_engine() -> None:
    assert SCHEDULER.actor_name == "scheduler"
    assert SCHEDULER.instance_id == "scheduler"
    assert identify("scheduler") == SCHEDULER


def test_a_bare_name_with_an_eight_digit_index_is_that_worker() -> None:
    assert identify("worker-12345678") == WorkerId(12345678)


@pytest.mark.parametrize(
    "value",
    [
        None,
        42,
        "",
        "scaler",
        "scheduler-0a1b2c3d",
        "worker",
        "worker-",
        "worker-01",
        "worker--1",
        "worker-1-0a1b2c3",
        "worker-1-0a1b2c3d-0a1b2c3d",
        "worker-1-0A1B2C3D",
        "worker-1 ",
        # named before workers spread over the nodes, after their node
        "worker-aaaaaaaaaaaa-0",
        "worker-123456789012-0",
    ],
)
def test_other_values_are_no_identity(value: object) -> None:
    assert identify(value) is None


@pytest.mark.parametrize("index", [-1, "3", 1.0, True])
def test_no_worker_identity_outside_the_naming_scheme(index: object) -> None:
    with pytest.raises(ValueError, match="worker"):
        WorkerId(index)  # type: ignore[arg-type]
