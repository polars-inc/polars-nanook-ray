import pathlib
import typing

import pytest
import toml

from polars_nanook_ray._engine import (
    SchedulerAddress,
    render_scheduler,
    render_worker,
)
from polars_nanook_ray._spec import build_spec
from polars_nanook_ray.options import ServiceAccount

if typing.TYPE_CHECKING:
    from polars_nanook_ray._spec import ClusterSpec

SCHEDULER_PORTS = {
    "client": 5051,
    "registration": 5050,
    "dashboard": 3001,
    "telemetry": 5049,
}
LEADER = SchedulerAddress(host="10.0.0.1", registration_port=5050, telemetry_port=5049)


def _scheduler(spec: "ClusterSpec", **kwargs: typing.Any) -> dict:
    kwargs.setdefault("ports", SCHEDULER_PORTS)
    return toml.loads(render_scheduler(spec, host="10.0.0.1", **kwargs))


def _worker(spec: "ClusterSpec") -> dict:
    return toml.loads(
        render_worker(
            spec,
            instance_id="worker-0",
            host="10.0.0.2",
            task_port=6000,
            shuffle_port=7000,
            scheduler=LEADER,
        )
    )


@pytest.mark.parametrize(
    "license_",
    ["https://license.example:50051", "http://license.example:50051"],
)
def test_license_server_uri_reaches_scheduler_and_workers(license_: str) -> None:
    spec = build_spec(license=license_)

    expected = {"license_server": {"uri": license_}}
    assert _scheduler(spec)["license"] == expected
    assert _worker(spec)["license"] == expected


@pytest.mark.parametrize(
    "license_",
    ["license.json", pathlib.Path("license.json"), "file://license.json"],
)
def test_license_file_made_absolute(license_: str | pathlib.Path) -> None:
    spec = build_spec(license=license_)

    expected = {
        "on_prem_enterprise": {"license_path": str(pathlib.Path.cwd() / "license.json")}
    }
    assert _scheduler(spec)["license"] == expected
    assert _worker(spec)["license"] == expected


def test_license_file_uri_keeps_absolute_path() -> None:
    spec = build_spec(license="file:///etc/polars/license.json")

    assert _scheduler(spec)["license"] == {
        "on_prem_enterprise": {"license_path": "/etc/polars/license.json"}
    }


@pytest.mark.parametrize(
    "license_",
    [
        ServiceAccount(workspace_id="ws", client_id="id", client_secret="secret"),
        {"workspace_id": "ws", "client_id": "id", "client_secret": "secret"},
    ],
)
def test_service_account_validates_on_the_scheduler_only(
    license_: ServiceAccount | dict,
) -> None:
    spec = build_spec(license=license_)

    assert _scheduler(spec)["license"] == {
        "on_prem": {
            "workspace_id": "ws",
            "client_id": "id",
            "client_secret": "secret",
            "cert_dir": "/tmp/polars/license",
        }
    }
    assert "license" not in _worker(spec)


def test_service_account_secret_hidden_from_repr() -> None:
    account = ServiceAccount(workspace_id="ws", client_id="id", client_secret="hush")

    assert "hush" not in repr(account)
    assert "hush" not in repr(build_spec(license=account))


@pytest.mark.parametrize("license_", ["ftp://license.example/license.json", 42])
def test_unsupported_license_rejected(license_: typing.Any) -> None:
    with pytest.raises(ValueError, match="license"):
        build_spec(license=license_)
