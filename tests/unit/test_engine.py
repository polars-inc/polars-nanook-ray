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
from polars_nanook_ray.options import ServiceAccount, Storage

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


@pytest.mark.parametrize(
    ("url", "section"),
    [
        ("s3://bucket/shuffle", "s3"),
        ("s3a://bucket/shuffle", "s3"),
        ("gs://bucket/shuffle", "gcs"),
        ("gcs://bucket/shuffle", "gcs"),
        ("az://container/shuffle", "abs"),
        ("azure://container/shuffle", "abs"),
        ("abfs://container/shuffle", "abs"),
        ("abfss://container@account.dfs.core.windows.net/shuffle", "abs"),
    ],
)
def test_scheme_picks_object_store(url: str, section: str) -> None:
    spec = build_spec(license="/license.json", shuffle_data=url)

    assert _worker(spec)["worker"]["shuffle_location"] == {section: {"url": url}}


def test_object_store_settings_rendered() -> None:
    spec = build_spec(
        license="/license.json",
        anonymous_results=Storage(
            "s3://bucket/results",
            options={"aws_region": "eu-west-1"},
            endpoint="http://minio:9000",
            presign_duration="5 secs",
            allow_deletes=False,
        ),
    )

    assert _scheduler(spec)["scheduler"]["anonymous_result_location"] == {
        "s3": {
            "url": "s3://bucket/results",
            "aws_region": "eu-west-1",
            "external_endpoint": "http://minio:9000",
            "presign_duration": "5 secs",
            "allow_deletes": False,
        }
    }


def test_paths_are_local_unless_shared() -> None:
    spec = build_spec(
        license="/license.json",
        shuffle_data=pathlib.Path("/mnt/local/shuffle"),
        checkpoints=Storage("/mnt/nfs/checkpoints", shared=True),
    )

    worker = _worker(spec)["worker"]
    assert worker["shuffle_location"] == {"local": {"path": "/mnt/local/shuffle"}}
    assert worker["checkpoint_location"] == {
        "shared_filesystem": {"path": "/mnt/nfs/checkpoints"}
    }


def test_home_expanded_in_paths() -> None:
    spec = build_spec(license="/license.json", shuffle_data="~/shuffle")

    assert _worker(spec)["worker"]["shuffle_location"] == {
        "local": {"path": str(pathlib.Path.home() / "shuffle")}
    }


def test_checkpoints_off_by_default() -> None:
    assert (
        "checkpoint_location"
        not in _worker(build_spec(license="/license.json"))["worker"]
    )


def test_storage_defaults_on_local_disk() -> None:
    spec = build_spec(license="/license.json")

    assert _scheduler(spec)["scheduler"]["anonymous_result_location"] == {
        "local": {"path": "/tmp/polars/anonymous-results"}
    }
    assert _worker(spec)["worker"]["shuffle_location"] == {
        "local": {"path": "/tmp/polars/shuffle-data"}
    }


@pytest.mark.parametrize(
    ("storage", "match"),
    [
        ("relative/shuffle", "absolute"),
        ("hdfs://namenode/shuffle", "unknown scheme 'hdfs'"),
        ({"url": "gs://bucket/x", "endpoint": "http://minio"}, "S3 only"),
        ({"url": "s3://bucket/x", "shared": True}, "paths only"),
        ({"url": "/mnt/x", "options": {"k": "v"}}, "object stores only"),
    ],
)
def test_invalid_storage_rejected(storage: typing.Any, match: str) -> None:
    with pytest.raises(ValueError, match=rf"(?s)shuffle_data.*{match}"):
        build_spec(license="/license.json", shuffle_data=storage)


def test_invalid_storage_rejected_on_construction() -> None:
    with pytest.raises(ValueError, match="absolute"):
        Storage("relative/shuffle")


@pytest.mark.parametrize(
    "keyword",
    ["license", "shuffle_data", "ports", "tls", "auth"],
)
def test_unknown_key_in_mapping_rejected(keyword: str) -> None:
    valid: dict[str, typing.Any] = {
        "license": {"workspace_id": "ws", "client_id": "id", "client_secret": "s"},
        "shuffle_data": {"url": "s3://bucket/shuffle"},
        "ports": {"client": 5051},
        "tls": {"cert_file": "/cert.pem", "key_file": "/key.pem"},
        "auth": {"url": "https://idp/jwks", "issuer": "idp", "audience": "aud"},
    }
    keywords = {"license": "/license.json"} | {keyword: valid[keyword] | {"typo": 1}}

    with pytest.raises(ValueError, match=rf"(?s){keyword}.*typo"):
        build_spec(**keywords)
