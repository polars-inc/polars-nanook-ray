import json
import time
import urllib.error
import urllib.request

import pytest
import ray

from polars_nanook_ray import Autoscale, RayComputeContext
from polars_nanook_ray._engine import SCALER_HOST
from polars_nanook_ray.identity import WorkerId, list_workers

from .conftest import ContextFactory, TestQuery, name_of, rescale, scheduler_actor


def _wait_for_workers(
    namespace: str,
    expected: int,
    timeout: float = 30,
) -> set[WorkerId]:
    deadline = time.monotonic() + timeout
    workers: set[WorkerId] = set()

    while time.monotonic() < deadline:
        workers = list_workers(namespace)
        if len(workers) == expected:
            return workers
        time.sleep(1)

    msg = f"Expected {expected} workers, got {sorted(workers)} after {timeout}s"
    raise AssertionError(msg)


def _workers_named(*indices: int) -> set[WorkerId]:
    return {WorkerId(index) for index in indices}


def _names(workers: set[WorkerId]) -> set[str]:
    return {worker.actor_name for worker in workers}


def test_upscale(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=1)
    namespace = name_of(cluster)

    assert _wait_for_workers(namespace, 1) == _workers_named(0)

    rescale(cluster, 2)

    assert _wait_for_workers(namespace, 2) == _workers_named(0, 1)


def test_downscale_delete(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=2)
    namespace = name_of(cluster)

    before = _wait_for_workers(namespace, 2)
    delete = set(sorted(before)[:1])
    rescale(cluster, 1, delete=_names(delete))
    after = _wait_for_workers(namespace, 1)

    assert after == before - delete


def test_downscale_keep(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=2)
    namespace = name_of(cluster)

    before = _wait_for_workers(namespace, 2)
    keep = set(sorted(before)[:1])
    rescale(cluster, 1, keep=_names(keep))
    after = _wait_for_workers(namespace, 1)

    assert after == keep


def test_downscale_to_zero(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=2)
    namespace = name_of(cluster)

    assert len(_wait_for_workers(namespace, 2)) == 2

    rescale(cluster, 0)
    assert len(_wait_for_workers(namespace, 0)) == 0


def test_freed_index_reused(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=2)
    namespace = name_of(cluster)

    assert _wait_for_workers(namespace, 2) == _workers_named(0, 1)

    # remove the first only, and scale back up: the new worker takes the freed index
    rescale(cluster, 1, keep=_names(_workers_named(1)))
    _wait_for_workers(namespace, 1)

    rescale(cluster, 2)

    assert _wait_for_workers(namespace, 2) == _workers_named(0, 1)


def test_repeated_upscale_adds_no_extra_workers(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=1)
    namespace = name_of(cluster)

    rescale(cluster, 3)
    rescale(cluster, 3)

    assert _wait_for_workers(namespace, 3) == _workers_named(0, 1, 2)


def _scaler_port(cluster: RayComputeContext) -> int | None:
    return ray.get(scheduler_actor(cluster).scaler_port.remote(), timeout=30)


def test_scaling_disabled(ray_cluster: ContextFactory) -> None:
    cluster = ray_cluster(workers=1)

    assert _scaler_port(cluster) is None


@pytest.mark.parametrize("max_workers", [8, None])
def test_scale_config(
    ray_cluster: ContextFactory,
    max_workers: int | None,
) -> None:
    cluster = ray_cluster(
        autoscale=Autoscale(min_workers=0, max_workers=max_workers),
        workers=1,
    )

    port = _scaler_port(cluster)
    url = f"http://{SCALER_HOST}:{port}/scale_config"

    with urllib.request.urlopen(url) as response:
        body = json.loads(response.read())

    assert body["desired"] == 1
    assert body["available"] == 1
    assert body["min"] == 0
    assert body["max"] == max_workers


def test_client_autoscaling(
    ray_cluster: ContextFactory,
    run_query: TestQuery,
) -> None:
    # the engine reaches the scheduler actor on a port picked free
    cluster = ray_cluster(autoscale=Autoscale(), workers=0)

    run_query(cluster, min_workers=2, max_workers=2)

    assert len(list_workers(name_of(cluster))) == 2
