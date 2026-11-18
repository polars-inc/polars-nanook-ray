import contextlib
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from polars_nanook_ray import subprocess_runner as runner_module
from polars_nanook_ray.identity import WorkerId
from polars_nanook_ray.subprocess_runner import EngineStartError, SubprocessRunner


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
    return SubprocessRunner(str(binary), WorkerId(0))


def test_actor_stays_when_its_binary_is_stopped(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    runner = _runner(tmp_path, "exec sleep 60")
    runner.start("", {})
    runner.stop()

    assert not exits.done.wait(timeout=1)


_STARTS_AN_ENGINE = """
import os, sys, pathlib
from polars_nanook_ray.identity import WorkerId
from polars_nanook_ray.subprocess_runner import SubprocessRunner

runner = SubprocessRunner(sys.argv[1], WorkerId(0))
runner.start("", {})
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

    runner.start("", {})
    try:
        assert runner.is_alive()
    finally:
        runner.stop()


LICENSE_ERROR = (
    "License file check failed: Failed to read license file: No such file or "
    "directory (os error 2)"
)
# as the engine logs a fatal error: a JSON line on stdout, a plain one on stderr
LOGS_LICENSE_ERROR = (
    'echo \'{"fields":{"message":"' + LICENSE_ERROR + "\\\\n\\\\nStack backtrace:"
    '\\\\n   0: <unknown>","error":"' + LICENSE_ERROR + '"},"level":"ERROR"}\'\n'
    "echo '\"" + LICENSE_ERROR + "\"' >&2"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _listens(port: int) -> str:
    server = (
        "import socket, time; s = socket.socket(); "
        f"s.bind(('', {port})); s.listen(); time.sleep(60)"
    )
    return f'exec {sys.executable} -c "{server}"'


def test_engine_output_still_reaches_the_actor_log(
    tmp_path: pathlib.Path,
    capfd: pytest.CaptureFixture[str],
    exits: Exits,  # noqa: ARG001
) -> None:
    runner = _runner(tmp_path, "echo to-stdout; echo to-stderr >&2; exit 1")
    runner.start("", {})

    with pytest.raises(EngineStartError):
        runner.wait_until_listening(_free_port(), timeout=10)
    out, err = capfd.readouterr()
    assert "to-stdout" in out
    assert "to-stderr" in err


def test_wait_returns_once_the_engine_listens(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    port = _free_port()
    runner = _runner(tmp_path, f"sleep 0.5\n{_listens(port)}")
    runner.start("", {})
    try:
        runner.wait_until_listening(port, timeout=10)

        assert runner.is_ready(port)
    finally:
        runner.stop()
    assert exits.codes == []


def test_engine_exiting_while_starting_raises_its_error(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    runner = _runner(tmp_path, f"{LOGS_LICENSE_ERROR}\nexit 1")
    runner.start("", {})

    with pytest.raises(EngineStartError) as raised:
        runner.wait_until_listening(_free_port(), timeout=10)

    assert str(raised.value) == (
        f"The worker-0 engine exited with code 1 while starting: {LICENSE_ERROR}"
    )
    # failing the actor's creation, which Ray does not retry, rather than its process
    assert not exits.done.wait(timeout=1)


def test_engine_never_listening_times_out_with_its_logged_errors(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    port = _free_port()
    runner = _runner(tmp_path, f"{LOGS_LICENSE_ERROR}\nexec sleep 60")
    runner.start("", {})
    try:
        with pytest.raises(EngineStartError) as raised:
            runner.wait_until_listening(port, timeout=1)
    finally:
        runner.stop()

    assert str(raised.value) == (
        f"Did not start within 1s: worker-0, whose engine does not "
        f"listen on port {port}. Its engine logged: {LICENSE_ERROR}"
    )
    assert exits.codes == []


def test_engine_exiting_after_it_listened_exits_the_actor(
    tmp_path: pathlib.Path, exits: Exits
) -> None:
    port = _free_port()
    runner = _runner(tmp_path, f"{_listens(port)} &\nsleep 2\nexit 3")
    runner.start("", {})
    runner.wait_until_listening(port, timeout=10)

    assert exits.done.wait(timeout=10)
    assert exits.codes == [1]
