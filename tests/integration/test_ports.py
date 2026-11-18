import socket
import time

import psutil
import pytest
import ray

from polars_nanook_ray import Ports

from .conftest import ContextFactory, worker_actors


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _listening_ports(pid: int) -> set[int]:
    return {
        conn.laddr.port
        for conn in psutil.Process(pid).net_connections(kind="tcp")
        if conn.status == psutil.CONN_LISTEN
    }


def test_workers_on_one_host_listen_on_distinct_ports(
    ray_cluster: ContextFactory,
) -> None:
    cluster = ray_cluster(workers=2)

    ports = [
        _listening_ports(ray.get(actor.get_worker_pid.remote()))
        for actor in worker_actors(cluster).values()
    ]

    assert all(ports)
    assert ports[0].isdisjoint(ports[1])


def test_worker_listens_on_fixed_port(ray_cluster: ContextFactory) -> None:
    port = _free_port()
    cluster = ray_cluster(
        workers=1, ports=Ports(client=None, dashboard=None, task=port)
    )

    (actor,) = worker_actors(cluster).values()
    pid = ray.get(actor.get_worker_pid.remote())

    assert port in _listening_ports(pid)


def _assert_start_fails_fast_on_taken_port(
    ray_cluster: ContextFactory,
    port_name: str,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen()
        port = listener.getsockname()[1]

        started = time.monotonic()
        with pytest.raises(Exception, match=rf"{port} is already in use"):
            ray_cluster(
                workers=1,
                ports=Ports(**{"client": None, "dashboard": None, port_name: port}),
            )

    assert time.monotonic() - started < 15


def test_taken_worker_port_fails_fast(ray_cluster: ContextFactory) -> None:
    _assert_start_fails_fast_on_taken_port(ray_cluster, "task")


def test_taken_scheduler_port_fails_fast(ray_cluster: ContextFactory) -> None:
    _assert_start_fails_fast_on_taken_port(ray_cluster, "client")
