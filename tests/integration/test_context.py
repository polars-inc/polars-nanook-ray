import urllib.error
import urllib.request

import pytest
import ray

from polars_nanook_ray import Ports, RayComputeContext

from .conftest import ContextFactory, TestQuery, make_context, name_of


def test_ray_compute_context(run_query: TestQuery) -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        with make_context(workers=1) as ctx:
            result = run_query(ctx)
    finally:
        ray.shutdown()

    assert result is not None


def test_connect_from_another_driver(
    ray_cluster: ContextFactory, run_query: TestQuery
) -> None:
    ctx = ray_cluster(workers=1)
    client_addr = ctx.client_addr

    ray.shutdown()  # the driver that started it exits, the cluster stays up
    ray.init(address="auto")

    connected = RayComputeContext.connect(name_of(ctx))

    assert connected.client_addr == client_addr
    assert run_query(connected) is not None

    connected.stop()
    with pytest.raises(RuntimeError, match=r"not .*running"):
        connected.client_addr


def test_observatory_on_free_ports_answers_queries(
    ray_cluster: ContextFactory, run_query: TestQuery
) -> None:
    # registration and telemetry ports picked free on the scheduler's node
    ctx = ray_cluster(
        workers=1,
        observatory=True,
        ports=Ports(client=None, dashboard=None),
    )

    assert run_query(ctx) is not None

    assert ctx.dashboard_addr is not None
    try:
        with urllib.request.urlopen(ctx.dashboard_addr, timeout=10) as response:
            status = response.status
    except urllib.error.HTTPError as exc:  # served, if not at the root
        status = exc.code
    assert status < 500
