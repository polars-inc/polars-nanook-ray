import os
import pathlib
import threading

import pytest

from polars_nanook_ray import subprocess_runner as runner_module
from polars_nanook_ray.subprocess_runner import SubprocessRunner


class Exits:
    def __init__(self) -> None:
        self.codes: list[int] = []
        self.done = threading.Event()

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        self.done.set()


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> Exits:
    exits = Exits()
    monkeypatch.setattr(runner_module.os, "_exit", exits)
    monkeypatch.setattr(runner_module.logging, "shutdown", lambda: None)
    return exits


def _runner(tmp_path: pathlib.Path, script: str) -> SubprocessRunner:
    binary = tmp_path / "binary"
    binary.write_text(f"#!/bin/sh\n{script}\n")
    binary.chmod(0o755)
    return SubprocessRunner(str(binary), "worker-0", "worker-0")


def test_actor_exits_when_its_binary_exits_unasked(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    _runner(tmp_path, "exit 3").start(tmp_path / "config.toml", dict(os.environ))

    assert exits.done.wait(timeout=10)
    assert exits.codes == [1]


def test_actor_stays_when_its_binary_is_stopped(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    runner = _runner(tmp_path, "exec sleep 60")
    runner.start(tmp_path / "config.toml", dict(os.environ))
    runner.stop()

    assert not exits.done.wait(timeout=1)
