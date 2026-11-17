import re

import pytest
import ray

from polars_nanook_ray.identity import SCHEDULER, WorkerId, identify


def _real_worker_ids() -> list[WorkerId]:
    node_ids = [ray.NodeID.from_random().hex() for _ in range(3)]
    return [WorkerId.of(node_id, index) for node_id in node_ids for index in (0, 1, 10)]


def test_worker_actor_name_is_its_node_prefix_and_index() -> None:
    assert WorkerId.of("a" * 56, 0).actor_name == "worker-aaaaaaaaaaaa-0"
    assert WorkerId.of("0123456789ab" + "c" * 44, 12).actor_name == (
        "worker-0123456789ab-12"
    )


def test_worker_instance_id_is_its_actor_name_and_a_start() -> None:
    instance_id = WorkerId.of("a" * 56, 3).new_instance_id()

    assert re.fullmatch(r"worker-aaaaaaaaaaaa-3-[0-9a-f]{8}", instance_id)


def test_each_start_gets_its_own_instance_id() -> None:
    worker = WorkerId.of("a" * 56, 0)

    assert worker.new_instance_id() != worker.new_instance_id()


@pytest.mark.parametrize("worker", _real_worker_ids(), ids=str)
def test_worker_names_and_instance_ids_resolve_to_the_worker(worker: WorkerId) -> None:
    assert identify(worker.actor_name) == worker
    assert identify(worker.new_instance_id()) == worker


def test_scheduler_is_named_scheduler_on_ray_and_to_the_engine() -> None:
    assert SCHEDULER.actor_name == "scheduler"
    assert SCHEDULER.instance_id == "scheduler"
    assert identify("scheduler") == SCHEDULER


def test_a_bare_name_with_an_eight_digit_index_is_that_worker() -> None:
    assert identify("worker-aaaaaaaaaaaa-12345678") == WorkerId(
        "aaaaaaaaaaaa", 12345678
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        42,
        "",
        "scaler",
        "scheduler-0a1b2c3d",
        "worker-1",
        "worker-aaaaaaaaaaa-0",  # 11-char node prefix
        "worker-aaaaaaaaaaaaa-0",  # 13
        "worker-AAAAAAAAAAAA-0",
        "worker-aaaaaaaaaaaa-01",
        "worker-aaaaaaaaaaaa-0-0a1b2c3",
        "worker-aaaaaaaaaaaa-0-0a1b2c3d-0a1b2c3d",
        "worker-aaaaaaaaaaaa-0 ",
    ],
)
def test_other_values_are_no_identity(value: object) -> None:
    assert identify(value) is None


@pytest.mark.parametrize(
    ("node", "index"),
    [
        ("a" * 11, 0),
        ("A" * 12, 0),
        ("a-" * 6, 0),
        ("a" * 12, -1),
        ("a" * 12, "3"),
        (123456789012, 0),
    ],
)
def test_no_worker_identity_outside_the_naming_scheme(
    node: object, index: object
) -> None:
    with pytest.raises(ValueError, match="worker"):
        WorkerId(node, index)  # type: ignore[arg-type]
