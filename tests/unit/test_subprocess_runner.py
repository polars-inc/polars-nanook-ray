import contextlib
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import pytest

from polars_nanook_ray import subprocess_runner as runner_module
from polars_nanook_ray.identity import WorkerId
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
    return SubprocessRunner(str(binary), WorkerId.of("a" * 56, 0))


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


_STARTS_AN_ENGINE = """
import os, sys, pathlib
from polars_nanook_ray.identity import WorkerId
from polars_nanook_ray.subprocess_runner import SubprocessRunner

runner = SubprocessRunner(sys.argv[1], WorkerId.of("a" * 56, 0))
runner.start(pathlib.Path(sys.argv[2]), dict(os.environ))
print(runner.pid(), flush=True)
sys.stdin.read()
"""


def _alive(pid: int) -> bool:
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):  # reaped
        return False
    # an exited child the kernel hasn't reaped yet is gone for our purposes
    return stat.split(")")[-1].split()[0] != "Z"


@pytest.mark.skipif(
    sys.platform != "linux", reason="parent-death signals are Linux-only"
)
def test_binary_dies_when_its_actor_process_is_killed(tmp_path: pathlib.Path) -> None:
    binary = tmp_path / "binary"
    binary.write_text("#!/bin/sh\nexec sleep 60\n")
    binary.chmod(0o755)
    actor = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _STARTS_AN_ENGINE,
            str(binary),
            str(tmp_path / "c.toml"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert actor.stdout is not None
    engine_pid = int(actor.stdout.readline())
    try:
        actor.kill()  # SIGKILL: no handler of ours runs
        actor.wait()

        deadline = time.monotonic() + 5
        while _alive(engine_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(engine_pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(engine_pid, signal.SIGKILL)


def test_binary_starts_without_prctl(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    exits: Exits,  # noqa: ARG001
) -> None:
    # e.g. a statically linked Python, whose C library exports no prctl
    monkeypatch.setattr(runner_module.ctypes, "CDLL", lambda *_, **__: object())
    runner = _runner(tmp_path, "exec sleep 60")

    runner.start(tmp_path / "config.toml", dict(os.environ))
    try:
        assert runner.is_alive()
    finally:
        runner.stop()
