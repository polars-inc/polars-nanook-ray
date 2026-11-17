import os
import subprocess
import sys

_LOGS = """
import logging, sys
{setup}
from polars_nanook_ray.actors.utils import _log_to_stderr

_log_to_stderr()
logging.getLogger("polars_nanook_ray.actors.worker").info("worker started")
"""


_ENV = {key: value for key, value in os.environ.items() if key != "LOGLEVEL"}


def _stderr(setup: str = "", **env: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", _LOGS.format(setup=setup)],
        capture_output=True,
        text=True,
        check=True,
        env={**_ENV, **env},
    ).stderr


def test_importing_the_package_leaves_logging_alone() -> None:
    root = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging, polars_nanook_ray; "
            "print(logging.getLogger().level, logging.getLogger().handlers)",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=_ENV,
    ).stdout

    assert root.strip() == "30 []"


def test_actors_log_info_to_stderr() -> None:
    assert "worker started" in _stderr()


def test_actors_honour_loglevel() -> None:
    assert "worker started" not in _stderr(LOGLEVEL="WARNING")


def test_actors_keep_a_logging_setup_of_their_own() -> None:
    setup = "logging.basicConfig(level=logging.WARNING, stream=sys.stdout)"

    assert "worker started" not in _stderr(setup)
