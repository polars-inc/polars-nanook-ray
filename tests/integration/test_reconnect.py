import ray
from polars.testing import assert_frame_equal

from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.identity import SCHEDULER, list_workers

from .conftest import RayClusterConfigFactory, TestQuery


def test_reconnect_existing_cluster(
    ray_cluster_config: RayClusterConfigFactory,
    run_query: TestQuery,
) -> None:
    config = ray_cluster_config(num_workers=1)

    ray.init(address="auto", namespace=config.cluster_id, ignore_reinit_error=True)
    cluster0 = PolarsRayCluster(config)
    cluster0.start()

    scheduler0 = ray.get_actor(SCHEDULER.actor_name, namespace=config.cluster_id)
    workers0 = [
        ray.get_actor(worker.actor_name, namespace=config.cluster_id)
        for worker in sorted(list_workers(config.cluster_id))
    ]
    result0 = run_query(cluster=cluster0)

    ray.shutdown()  # simulates the calling process exiting (cluster stays up)

    ray.init(address="auto", namespace=config.cluster_id, ignore_reinit_error=True)
    cluster1 = PolarsRayCluster(config)
    cluster1.start()

    scheduler1 = ray.get_actor(SCHEDULER.actor_name, namespace=config.cluster_id)
    workers1 = [
        ray.get_actor(worker.actor_name, namespace=config.cluster_id)
        for worker in sorted(list_workers(config.cluster_id))
    ]
    result1 = run_query(cluster=cluster1)

    try:
        assert scheduler1 == scheduler0
        assert workers1 == workers0

        assert result0 is not None
        assert result1 is not None
        assert_frame_equal(result0, result1, check_row_order=False)

        cluster1.stop()
    finally:
        ray.shutdown()


def test_reconnect_after_upscale_keeps_workers(
    ray_cluster_config: RayClusterConfigFactory,
) -> None:
    config = ray_cluster_config(num_workers=1)

    ray.init(address="auto", namespace=config.cluster_id, ignore_reinit_error=True)
    try:
        cluster0 = PolarsRayCluster(config)
        cluster0.start()
        cluster0._rescale_worker_pool_to(2)
        workers0 = list_workers(config.cluster_id)

        cluster1 = PolarsRayCluster(config)
        cluster1.start()

        assert list_workers(config.cluster_id) == workers0
        assert set(cluster1._worker_actors) == workers0

        cluster1.stop()
    finally:
        ray.shutdown()
