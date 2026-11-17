import pytest

from polars_nanook_ray.actors import placement
from polars_nanook_ray.actors.placement import (
    NoRoomForWorkers,
    check_room,
    plan_workers,
)
from polars_nanook_ray.identity import WorkerId

GIB = 1024**3


def test_plan_takes_the_lowest_free_indices() -> None:
    assert plan_workers(3, set()) == [WorkerId(0), WorkerId(1), WorkerId(2)]


def test_plan_reuses_a_freed_index() -> None:
    assert plan_workers(3, {WorkerId(1)}) == [WorkerId(0), WorkerId(2)]


@pytest.mark.parametrize("target", [0, 1, -1])
def test_plan_nothing_when_the_workers_exist(target: int) -> None:
    assert plan_workers(target, {WorkerId(0)}) == []


def _nodes(monkeypatch: pytest.MonkeyPatch, *nodes: tuple[float, float, bool]) -> None:
    monkeypatch.setattr(
        placement.ray,
        "nodes",
        lambda: [
            {"Alive": alive, "Resources": {"CPU": cpus, "memory": memory}}
            for cpus, memory, alive in nodes
        ],
    )


def test_room_on_a_node_that_fits_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _nodes(monkeypatch, (2, 4 * GIB, True), (8, 16 * GIB, True))

    check_room(cpus=4, memory=8 * GIB)


@pytest.mark.parametrize(
    ("cpus", "memory", "match"),
    [
        (16, GIB, "requests 16 CPU, but the largest node has 8 CPU"),
        (1, 32 * GIB, "requests 32 GiB of memory, but the largest node has 16 GiB"),
        (4, 8 * GIB, "requests 4 CPU and 8 GiB of memory, but no single node has both"),
    ],
)
def test_no_room_when_a_worker_fits_no_node(
    monkeypatch: pytest.MonkeyPatch, cpus: int, memory: int, match: str
) -> None:
    _nodes(monkeypatch, (2, 16 * GIB, True), (8, 4 * GIB, True))

    with pytest.raises(NoRoomForWorkers, match=match):
        check_room(cpus=cpus, memory=memory)


def test_dead_nodes_have_no_room(monkeypatch: pytest.MonkeyPatch) -> None:
    _nodes(monkeypatch, (8, 16 * GIB, False))

    with pytest.raises(NoRoomForWorkers):
        check_room(cpus=1, memory=GIB)
