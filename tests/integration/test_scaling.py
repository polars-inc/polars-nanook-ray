import json
import time
import urllib.request

import pytest
import ray

from polars_nanook_ray.actors import (
    SCALER_NAME_PREFIX,
    list_worker_names,
    worker_name,
)
from polars_nanook_ray.config import SCALER_HOST

from .conftest import RayClusterFactory, TestQuery


def _wait_for_workers(
    cluster_id: str,
    expected: int,
    timeout: float = 30,
) -> set[str]:
    deadline = time.monotonic() + timeout
    names: set[str] = set()

    while time.monotonic() < deadline:
        names = list_worker_names(cluster_id)
        if len(names) == expected:
            return names
        time.sleep(1)

    msg = f"Expected {expected} workers, got {sorted(names)} after {timeout}s"
    raise AssertionError(msg)


def _worker_names(*indices: int) -> set[str]:
    return {worker_name(index) for index in indices}


def test_upscale(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id

    names = _wait_for_workers(cluster_id, 1)
    assert names == _worker_names(0)

    cluster._rescale_worker_pool_to(2)

    names = _wait_for_workers(cluster_id, 2)
    assert names == _worker_names(0, 1)


def test_downscale_delete(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    before = _wait_for_workers(cluster_id, 2)
    delete = set(list(before)[:1])
    cluster._rescale_worker_pool_to(1, delete=delete)
    after = _wait_for_workers(cluster_id, 1)

    assert after == before - delete


def test_downscale_keep(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    before = _wait_for_workers(cluster_id, 2)
    keep = set(list(before)[:1])
    cluster._rescale_worker_pool_to(1, keep=keep)
    after = _wait_for_workers(cluster_id, 1)

    assert after == keep


def test_downscale_to_zero(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    assert len(_wait_for_workers(cluster_id, 2)) == 2

    cluster._rescale_worker_pool_to(0)
    assert len(_wait_for_workers(cluster_id, 0)) == 0


def test_freed_index_reused(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=2)
    cluster_id = cluster.config.cluster_id

    names = _wait_for_workers(cluster_id, 2)
    assert names == _worker_names(0, 1)

    # remove the first only, and scale back up: the new worker takes the freed index
    cluster._rescale_worker_pool_to(1, keep=_worker_names(1))
    _wait_for_workers(cluster_id, 1)

    cluster._rescale_worker_pool_to(2)

    names = _wait_for_workers(cluster_id, 2)
    assert names == _worker_names(0, 1)


def test_repeated_upscale_adds_no_extra_workers(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id

    cluster._rescale_worker_pool_to(3)
    cluster._rescale_worker_pool_to(3)

    assert _wait_for_workers(cluster_id, 3) == _worker_names(0, 1, 2)


def test_scaling_disabled(ray_cluster: RayClusterFactory) -> None:
    cluster = ray_cluster(
        scaling_enabled=False,
        num_workers=1,
    )

    with pytest.raises(ValueError, match=SCALER_NAME_PREFIX):
        ray.get_actor(SCALER_NAME_PREFIX, namespace=cluster.config.cluster_id)


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

    assert len(list_worker_names(cluster.config.cluster_id)) == 2
