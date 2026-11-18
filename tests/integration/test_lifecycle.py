import pathlib
import time

import pytest
import ray
from polars.testing import assert_frame_equal

from .conftest import ContextFactory, TestQuery, make_context


def test_two_sequential_polars_clusters_one_ray_cluster(run_query: TestQuery) -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        with make_context(workers=1) as ctx0:
            result0 = run_query(ctx0)

        with make_context(workers=1) as ctx1:
            result1 = run_query(ctx1)

        assert result0 is not None
        assert result1 is not None
        assert_frame_equal(result0, result1, check_row_order=False)
    finally:
        ray.shutdown()


def test_two_parallel_polars_clusters_one_ray_cluster(run_query: TestQuery) -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        with make_context(workers=1) as ctx0, make_context(workers=1) as ctx1:
            result0 = run_query(ctx0)
            result1 = run_query(ctx1)

        assert result0 is not None
        assert result1 is not None
        assert_frame_equal(result0, result1, check_row_order=False)
    finally:
        ray.shutdown()


def test_scheduler_exit_fails_fast(
    ray_cluster: ContextFactory,
    tmp_path: pathlib.Path,
) -> None:
    started = time.monotonic()
    # the engine's exit takes its actor down, and Ray retries it up to its limit
    with pytest.raises(RuntimeError, match="scheduler exited"):
        ray_cluster(workers=0, license=tmp_path / "missing.json")

    assert time.monotonic() - started < 15
