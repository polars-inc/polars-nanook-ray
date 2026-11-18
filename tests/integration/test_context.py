import urllib.error
import urllib.request

import pytest
import ray

from polars_nanook_ray import Ports, RayClusterContext

from .conftest import ContextFactory, TestQuery, make_context, name_of


def test_ray_cluster_context(run_query: TestQuery) -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    try:
        with make_context(workers=1) as ctx:
            result = run_query(ctx)
    finally:
        ray.shutdown()

    assert result is not None


def test_ray_init_needs_no_namespace(run_query: TestQuery) -> None:
    ray.init(address="auto")
    try:
        ctx = make_context(workers=1)
        ctx.start()
        try:
            assert ray.get_runtime_context().namespace != name_of(ctx)
            assert run_query(ctx) is not None
        finally:
            ctx.stop()
    finally:
        ray.shutdown()


def test_connect_from_another_driver(
    ray_cluster: ContextFactory, run_query: TestQuery
) -> None:
    ctx = ray_cluster(workers=1)
    client_addr = ctx.client_addr

    ray.shutdown()  # the driver that started it exits, the cluster stays up
    ray.init(address="auto")

    connected = RayClusterContext.connect(name_of(ctx))

    assert connected.client_addr == client_addr
    assert run_query(connected) is not None

    connected.stop()
    with pytest.raises(RuntimeError, match=r"not .*running"):
        connected.client_addr


def test_connect_without_a_cluster_names_it() -> None:
    ray.init(address="auto")
    try:
        with pytest.raises(ValueError, match="no-such-cluster"):
            RayClusterContext.connect("no-such-cluster")
    finally:
        ray.shutdown()


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


def test_no_dashboard_without_observatory(ray_cluster: ContextFactory) -> None:
    ctx = ray_cluster(workers=0)

    assert ctx.dashboard_addr is None
    assert ctx.client_addr.startswith("http://")
