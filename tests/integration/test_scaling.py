import json
import time
import urllib.error
import urllib.request

import pytest
import ray

from polars_nanook_ray.actors import WorkerSlot
from polars_nanook_ray.config import SCALER_HOST
from polars_nanook_ray.identity import WorkerId, list_workers

from .conftest import RayClusterFactory, TestQuery


def _wait_for_workers(
    cluster_id: str,
    expected: int,
    timeout: float = 30,
) -> set[WorkerId]:
    deadline = time.monotonic() + timeout
    workers: set[WorkerId] = set()

    while time.monotonic() < deadline:
        workers = list_workers(cluster_id)
        if len(workers) == expected:
            return workers
        time.sleep(1)

    msg = f"Expected {expected} workers, got {sorted(workers)} after {timeout}s"
    raise AssertionError(msg)


def _slots(*indices: int) -> set[WorkerId]:
    node_id = ray.get_runtime_context().get_node_id()
    return {WorkerSlot(node_id, index).worker_id for index in indices}


def _names(workers: set[WorkerId]) -> set[str]:
    return {worker.actor_name for worker in workers}


def test_upscale(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id

    assert _wait_for_workers(cluster_id, 1) == _slots(0)

    cluster._rescale_worker_pool_to(2)

    assert _wait_for_workers(cluster_id, 2) == _slots(0, 1)


def test_downscale_delete(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    before = _wait_for_workers(cluster_id, 2)
    delete = set(sorted(before)[:1])
    cluster._rescale_worker_pool_to(1, delete=_names(delete))
    after = _wait_for_workers(cluster_id, 1)

    assert after == before - delete


def test_downscale_keep(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    before = _wait_for_workers(cluster_id, 2)
    keep = set(sorted(before)[:1])
    cluster._rescale_worker_pool_to(1, keep=_names(keep))
    after = _wait_for_workers(cluster_id, 1)

    assert after == keep


def test_downscale_to_zero(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    assert len(_wait_for_workers(cluster_id, 2)) == 2

    cluster._rescale_worker_pool_to(0)
    assert len(_wait_for_workers(cluster_id, 0)) == 0


def test_freed_slot_reused(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    assert _wait_for_workers(cluster_id, 2) == _slots(0, 1)

    # remove the first only, and scale back up: the new worker takes the freed slot
    cluster._rescale_worker_pool_to(1, keep=_names(_slots(1)))
    _wait_for_workers(cluster_id, 1)

    cluster._rescale_worker_pool_to(2)

    assert _wait_for_workers(cluster_id, 2) == _slots(0, 1)


def test_repeated_upscale_adds_no_extra_workers(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id

    cluster._rescale_worker_pool_to(3)
    cluster._rescale_worker_pool_to(3)

    assert _wait_for_workers(cluster_id, 3) == _slots(0, 1, 2)


def test_scaling_disabled(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(
        scaling_enabled=False,
        num_workers=1,
    )

    port = cluster.config.scheduler.scaling.port
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://{SCALER_HOST}:{port}/scale_config")


@pytest.mark.parametrize("max_workers", [8, None])
def test_scale_config(
    ray_cluster: RayClusterFactory,
    max_workers: int | None,
) -> None:
    cluster = ray_cluster(
        scaling_enabled=True,
        num_workers=1,
        min_workers=0,
        max_workers=max_workers,
    )

    port = cluster.config.scheduler.scaling.port
    url = f"http://{SCALER_HOST}:{port}/scale_config"

    with urllib.request.urlopen(url) as response:
        body = json.loads(response.read())

    assert body["desired"] == 1
    assert body["available"] == 1
    assert body["min"] == 0
    assert body["max"] == max_workers


def test_client_autoscaling(
    ray_cluster: RayClusterFactory,
    run_query: TestQuery,
) -> None:
    cluster = ray_cluster(
        scaling_enabled=True,
        num_workers=0,
    )

    run_query(cluster=cluster, min_workers=2, max_workers=2)

    assert len(list_workers(cluster.config.cluster_id)) == 2
