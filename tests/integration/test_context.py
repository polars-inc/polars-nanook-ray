import ray

from .conftest import TestQuery, make_context


def test_ray_cluster_context(run_query: TestQuery) -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        with make_context(workers=1) as ctx:
            result = run_query(ctx)
    finally:
        ray.shutdown()

    assert result is not None
