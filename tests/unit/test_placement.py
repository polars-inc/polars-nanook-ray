import pytest

from polars_nanook_ray.actors.placement import (
    WorkerSlot,
    _node_capacities,
    _plan_worker_slots,
)

NODE_A = "a" * 56
NODE_B = "b" * 56
NODE_C = "c" * 56


def _node(node_id: str, *, alive: bool = True, **resources: float) -> dict:
    return {"NodeID": node_id, "Alive": alive, "Resources": resources}


def test_slot_worker_is_named_after_its_node_and_index() -> None:
    assert WorkerSlot(NODE_A, 0).worker_id.actor_name == "worker-aaaaaaaaaaaa-0"
    assert WorkerSlot(NODE_B, 3).worker_id.actor_name == "worker-bbbbbbbbbbbb-3"


def test_capacity_skips_dead_and_cpuless_nodes() -> None:
    nodes = [
        _node(NODE_C, CPU=4, memory=1e9),
        _node(NODE_A, CPU=0, memory=1e9),  # e.g. a head started with --num-cpus=0
        _node(NODE_B, alive=False, CPU=4, memory=1e9),
    ]

    assert _node_capacities(nodes, cpus=0, memory=0, workers_per_node=1) == {NODE_C: 1}


def test_capacity_skips_nodes_too_small_for_a_worker() -> None:
    nodes = [
        _node(NODE_A, CPU=2, memory=8e9),
        _node(NODE_B, CPU=8, memory=2e9),
        _node(NODE_C, CPU=8, memory=8e9),
    ]

    assert _node_capacities(nodes, cpus=4, memory=4e9, workers_per_node=1) == {
        NODE_C: 1
    }


def test_capacity_is_what_the_node_fits_up_to_workers_per_node() -> None:
    nodes = [
        _node(NODE_A, CPU=2, memory=8e9),
        _node(NODE_B, CPU=4, memory=8e9),
        _node(NODE_C, CPU=16, memory=3e9),
    ]

    assert _node_capacities(nodes, cpus=2, memory=1e9, workers_per_node=3) == {
        NODE_A: 1,
        NODE_B: 2,
        NODE_C: 3,
    }


def test_plan_follows_capacity_per_node() -> None:
    slots = _plan_worker_slots({NODE_A: 1, NODE_B: 2}, taken=set(), count=3)

    assert slots == [
        WorkerSlot(NODE_A, 0),
        WorkerSlot(NODE_B, 0),
        WorkerSlot(NODE_B, 1),
    ]


def test_plan_puts_one_worker_per_node() -> None:
    slots = _plan_worker_slots({NODE_A: 1, NODE_B: 1, NODE_C: 1}, taken=set(), count=2)

    assert slots == [WorkerSlot(NODE_A, 0), WorkerSlot(NODE_B, 0)]


def test_plan_skips_nodes_that_already_have_a_worker() -> None:
    taken = {WorkerSlot(NODE_A, 0).worker_id}

    slots = _plan_worker_slots({NODE_A: 1, NODE_B: 1, NODE_C: 1}, taken=taken, count=2)

    assert slots == [WorkerSlot(NODE_B, 0), WorkerSlot(NODE_C, 0)]


def test_plan_stops_at_capacity() -> None:
    slots = _plan_worker_slots({NODE_A: 1, NODE_B: 1}, taken=set(), count=5)

    assert slots == [WorkerSlot(NODE_A, 0), WorkerSlot(NODE_B, 0)]


def test_plan_spreads_before_stacking_when_several_workers_per_node() -> None:
    slots = _plan_worker_slots({NODE_A: 2, NODE_B: 2}, taken=set(), count=3)

    assert slots == [
        WorkerSlot(NODE_A, 0),
        WorkerSlot(NODE_B, 0),
        WorkerSlot(NODE_A, 1),
    ]


def test_plan_reuses_a_freed_slot() -> None:
    taken = {WorkerSlot(NODE_A, 1).worker_id}

    slots = _plan_worker_slots({NODE_A: 3}, taken=taken, count=2)

    assert slots == [WorkerSlot(NODE_A, 0), WorkerSlot(NODE_A, 2)]


@pytest.mark.parametrize("count", [0, -1])
def test_plan_nothing(count: int) -> None:
    assert _plan_worker_slots({NODE_A: 1}, taken=set(), count=count) == []
