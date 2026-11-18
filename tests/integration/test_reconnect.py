import ray
from polars.testing import assert_frame_equal

from polars_nanook_ray.identity import SCHEDULER, list_workers

from .conftest import TestQuery, make_context, name_of, rescale


def test_reconnect_existing_cluster(run_query: TestQuery) -> None:
    ctx0 = make_context(workers=1)
    name = name_of(ctx0)

    ray.init(address="auto", ignore_reinit_error=True)
    ctx0.start()

    scheduler0 = ray.get_actor(SCHEDULER.actor_name, namespace=name)
    workers0 = [
        ray.get_actor(worker.actor_name, namespace=name)
        for worker in sorted(list_workers(name))
    ]
    result0 = run_query(ctx0)

    ray.shutdown()  # simulates the calling process exiting (cluster stays up)

    ray.init(address="auto", ignore_reinit_error=True)
    ctx1 = make_context(name=name, workers=1)
    ctx1.start()

    scheduler1 = ray.get_actor(SCHEDULER.actor_name, namespace=name)
    workers1 = [
        ray.get_actor(worker.actor_name, namespace=name)
        for worker in sorted(list_workers(name))
    ]
    result1 = run_query(ctx1)

    try:
        assert scheduler1 == scheduler0
        assert workers1 == workers0

        assert result0 is not None
        assert result1 is not None
        assert_frame_equal(result0, result1, check_row_order=False)

        ctx1.stop()
    finally:
        ray.shutdown()


def test_reconnect_after_upscale_keeps_workers() -> None:
    ctx0 = make_context(workers=1)
    name = name_of(ctx0)

    ray.init(address="auto", ignore_reinit_error=True)
    try:
        ctx0.start()
        rescale(ctx0, 2)
        workers0 = list_workers(name)

        ctx1 = make_context(name=name, workers=1)
        ctx1.start()

        assert list_workers(name) == workers0

        ctx1.stop()
    finally:
        ray.shutdown()
