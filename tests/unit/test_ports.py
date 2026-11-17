import socket

import pytest

from polars_nanook_ray.actors.ports import PortInUseError, resolve_ports


def test_unset_ports_are_distinct_and_bindable() -> None:
    ports = resolve_ports({"task_port": None, "shuffle_port": None})

    assert ports["task_port"] != ports["shuffle_port"]
    for port in ports.values():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", port))


def test_fixed_port_is_kept() -> None:
    port = resolve_ports({"task_port": None})["task_port"]

    assert resolve_ports({"task_port": port, "shuffle_port": None})["task_port"] == port


def test_taken_fixed_port_raises() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        with pytest.raises(PortInUseError, match=rf"task_port {port} .*in use"):
            resolve_ports({"task_port": port})
