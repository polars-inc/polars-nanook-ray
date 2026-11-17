import socket
import time

import psutil
import pytest
import ray

from .conftest import RayClusterFactory, _free_port


def _listening_ports(pid: int) -> set[int]:
    return {
        conn.laddr.port
        for conn in psutil.Process(pid).net_connections(kind="tcp")
        if conn.status == psutil.CONN_LISTEN
    }


def test_workers_on_one_host_listen_on_distinct_ports(
    ray_cluster: RayClusterFactory,
) -> None:
    cluster = ray_cluster(num_workers=2)

    ports = [
        _listening_ports(ray.get(actor.get_worker_pid.remote()))
        for actor in cluster._worker_actors.values()
    ]

    assert all(ports)
    assert ports[0].isdisjoint(ports[1])


def test_worker_listens_on_fixed_port(ray_cluster: RayClusterFactory) -> None:
    port = _free_port()
    cluster = ray_cluster(num_workers=1, worker_task_port=port)

    pid = ray.get(next(iter(cluster._worker_actors.values())).get_worker_pid.remote())

    assert port in _listening_ports(pid)


def _assert_start_fails_fast_on_taken_port(
    ray_cluster: RayClusterFactory,
    port_kwarg: str,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen()
        port = listener.getsockname()[1]

        started = time.monotonic()
        with pytest.raises(Exception, match=rf"{port} is already in use"):
            ray_cluster(num_workers=1, **{port_kwarg: port})

    assert time.monotonic() - started < 15


def test_taken_worker_port_fails_fast(ray_cluster: RayClusterFactory) -> None:
    _assert_start_fails_fast_on_taken_port(ray_cluster, "worker_task_port")


def test_taken_scheduler_port_fails_fast(ray_cluster: RayClusterFactory) -> None:
    _assert_start_fails_fast_on_taken_port(ray_cluster, "scheduler_client_port")
