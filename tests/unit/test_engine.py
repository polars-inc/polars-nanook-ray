import pathlib
import typing

import pytest
import toml

from polars_nanook_ray._engine import (
    SchedulerAddress,
    render_scheduler,
    render_worker,
    scheduler_env,
    scheduler_ports,
    worker_dirs,
    worker_env,
    worker_ports,
)
from polars_nanook_ray._spec import build_spec
from polars_nanook_ray.options import (
    Autoscale,
    Jwks,
    OutOfCore,
    ServiceAccount,
    Storage,
    Tls,
)

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
        anonymous_results=Storage("/mnt/nfs/results", shared=True),
    )

    worker = _worker(spec)["worker"]
    assert worker["shuffle_location"] == {"local": {"path": "/mnt/local/shuffle"}}
    assert _scheduler(spec)["scheduler"]["anonymous_result_location"] == {
        "shared_filesystem": {"path": "/mnt/nfs/results"}
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


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"name": "my/cluster"}, "name"),
        ({"name": "-cluster"}, "name"),
        ({"workers": -1}, "workers"),
        ({"reconcile_interval": -1}, "reconcile_interval"),
        ({"temp_dir": "tmp"}, "temp_dir"),
        ({"out_of_core": {"spill_dir": "spill"}}, "spill_dir"),
        ({"out_of_core": {"memory_budget_fraction": 1.5}}, "memory_budget_fraction"),
        ({"out_of_core": {"disk_budget_mb": 0}}, "disk_budget_mb"),
        ({"out_of_core": {"spill_to": "/x"}}, "spill_to"),
        ({"log_level": "warn"}, "log_level"),
    ],
)
def test_invalid_keywords_rejected(keywords: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_spec(license="/license.json", **keywords)


def test_tls_and_auth_secure_client_port_and_dashboard() -> None:
    spec = build_spec(
        license="/license.json",
        tls=Tls(
            cert_file="/cert.pem",
            key_file="/key.pem",
            auth=Jwks(url="https://idp/jwks", issuer="https://idp", audience="aud"),
        ),
    )

    rendered = _scheduler(spec)
    expected = {
        "connection": {
            "tls": {"certificate_key_file": "/cert.pem", "private_key_file": "/key.pem"}
        },
        "auth": {
            "jwks": {
                "url": "https://idp/jwks",
                "issuers": "https://idp",
                "audience": "aud",
            }
        },
    }
    assert rendered["scheduler"]["client_service"] == {"bind_addr": ":5051"} | expected
    assert (
        rendered["observatory"]["rest_api"]["service"]
        == {"bind_addr": ":3001"} | expected
    )


def test_resolved_ports_bound_and_advertised() -> None:
    spec = build_spec(license="/license.json")

    scheduler = _scheduler(
        spec,
        ports={
            "client": 7001,
            "registration": 7002,
            "dashboard": 7003,
            "telemetry": 7004,
        },
    )
    worker = toml.loads(
        render_worker(
            spec,
            instance_id="worker-0",
            host="10.0.0.2",
            task_port=7005,
            shuffle_port=7006,
            scheduler=SchedulerAddress("10.0.0.1", 7002, 7004),
        )
    )

    assert scheduler["scheduler"]["client_service"]["bind_addr"] == ":7001"
    assert scheduler["scheduler"]["worker_service"]["bind_addr"] == ":7002"
    assert scheduler["observatory"]["rest_api"]["service"]["bind_addr"] == ":7003"
    assert scheduler["observatory"]["service"]["bind_addr"] == ":7004"
    leader = {
        "leader_instance_id": "scheduler",
        "scheduler_service": {"public_addr": "10.0.0.1:7002"},
        "observatory_service": {"public_addr": "10.0.0.1:7004"},
    }
    assert scheduler["static_leader"] == leader
    assert worker["static_leader"] == leader
    assert worker["worker"]["task_service"] == {
        "bind_addr": ":7005",
        "public_addr": "10.0.0.2:7005",
    }
    assert worker["worker"]["shuffle_service"] == {
        "bind_addr": ":7006",
        "public_addr": "10.0.0.2:7006",
    }


@pytest.mark.parametrize(
    ("ports", "scheduler", "worker"),
    [
        (
            None,
            {
                "client": 5051,
                "registration": None,
                "dashboard": 3001,
                "telemetry": None,
            },
            {"task": None, "shuffle": None},
        ),
        (
            {"client": None, "task": 6000, "shuffle": 6001, "telemetry": 4317},
            {
                "client": None,
                "registration": None,
                "dashboard": 3001,
                "telemetry": 4317,
            },
            {"task": 6000, "shuffle": 6001},
        ),
    ],
)
def test_ports_to_bind(ports: dict | None, scheduler: dict, worker: dict) -> None:
    spec = build_spec(license="/license.json", ports=ports)

    assert scheduler_ports(spec) == scheduler
    assert worker_ports(spec) == worker


@pytest.mark.parametrize(
    ("keywords", "per_query"),
    [
        ({}, {"max_workers_per_query": 1, "default_workers_per_query": 1}),
        ({"workers": 4}, {"max_workers_per_query": 4, "default_workers_per_query": 4}),
        (
            {"workers": 2, "autoscale": Autoscale(max_workers=8)},
            {"max_workers_per_query": 8, "default_workers_per_query": 2},
        ),
        (
            {"workers": 0, "autoscale": Autoscale(min_workers=3)},
            {"default_workers_per_query": 3},
        ),
        ({"workers": 0, "autoscale": Autoscale()}, {"default_workers_per_query": 1}),
        ({"workers": 0}, {}),
    ],
)
def test_workers_per_query(keywords: dict, per_query: dict) -> None:
    rendered = _scheduler(
        build_spec(license="/license.json", **keywords), scaler_port=4100
    )

    keys = ("max_workers_per_query", "default_workers_per_query")
    assert {k: v for k, v in rendered["scheduler"].items() if k in keys} == per_query


def test_autoscale_points_engine_at_scaler_port() -> None:
    spec = build_spec(license="/license.json", autoscale=Autoscale())

    rendered = _scheduler(spec, scaler_port=41234)

    assert rendered["scaling"] == {
        "enabled": True,
        "rest": {"uri": "http://127.0.0.1:41234"},
    }


@pytest.mark.parametrize(
    ("keywords", "resources"),
    [
        ({}, {"cpu_reserved": "1.0"}),
        ({"cpus": 0}, {}),
        ({"cpus": 2.5, "memory": 4}, {"cpu_reserved": "2.5"}),
    ],
)
def test_worker_resources(keywords: dict, resources: dict) -> None:
    spec = build_spec(license="/license.json", **keywords)

    worker = _worker(spec)
    assert {k: worker[k] for k in ("cpu_reserved", "memory_limit") if k in worker} == (
        resources
    )
    scheduler = _scheduler(spec)
    assert "cpu_reserved" not in scheduler
    assert "memory_limit" not in scheduler


def test_engine_environment() -> None:
    spec = build_spec(
        license="/license.json",
        log_level="trace",
        temp_dir="/scratch",
        env_vars={"RUST_BACKTRACE": "1", "FOO": "bar"},
    )

    expected = {
        "PLC_LOG_LEVEL": "trace",
        "RUST_BACKTRACE": "1",
        "POLARS_EULA_ACCEPTED": "yes",
        "FOO": "bar",
    }
    assert scheduler_env(spec) == expected
    assert worker_env(spec) == expected | {
        "POLARS_TEMP_DIR": "/scratch",
        "POLARS_OOC_SPILL_DIR": "/var/tmp/polars/spill",
        "POLARS_OVERRIDE_TOTAL_MEMORY_MB": "2147",
        "POLARS_OOMKILL_THRESHOLD_MB": "2048",
    }


def _ooc_env(spec: "ClusterSpec") -> dict[str, str]:
    return {
        k: v
        for k, v in worker_env(spec).items()
        if k.startswith(("POLARS_OOC_", "POLARS_OOMKILL_", "POLARS_OVERRIDE_"))
    }


def test_out_of_core_budgets_follow_worker_memory_by_default() -> None:
    spec = build_spec(license="/license.json", memory=8)

    assert _ooc_env(spec) == {
        "POLARS_OOC_SPILL_DIR": "/var/tmp/polars/spill",
        "POLARS_OVERRIDE_TOTAL_MEMORY_MB": "8589",
        "POLARS_OOMKILL_THRESHOLD_MB": "8192",
    }


def test_zero_memory_leaves_out_of_core_budgets_to_the_node() -> None:
    spec = build_spec(license="/license.json", memory=0)

    assert _ooc_env(spec) == {"POLARS_OOC_SPILL_DIR": "/var/tmp/polars/spill"}


@pytest.mark.parametrize(
    "out_of_core",
    [
        OutOfCore(
            spill_dir="~/spill",
            memory_budget_fraction=0.7,
            memory_budget_mb=1500,
            disk_budget_mb=200_000,
            oom_kill_threshold_mb=1900,
        ),
        {
            "spill_dir": "~/spill",
            "memory_budget_fraction": 0.7,
            "memory_budget_mb": 1500,
            "disk_budget_mb": 200_000,
            "oom_kill_threshold_mb": 1900,
        },
    ],
)
def test_out_of_core_settings_reach_workers_only(out_of_core: typing.Any) -> None:
    spec = build_spec(license="/license.json", out_of_core=out_of_core)

    spill_dir = str(pathlib.Path("~/spill").expanduser())
    assert _ooc_env(spec) == {
        "POLARS_OOC_SPILL_DIR": spill_dir,
        "POLARS_OVERRIDE_TOTAL_MEMORY_MB": "2147",
        "POLARS_OOMKILL_THRESHOLD_MB": "1900",
        "POLARS_OOC_MEMORY_BUDGET_FRACTION": "0.7",
        "POLARS_OOC_MEMORY_BUDGET_MB": "1500",
        "POLARS_OOC_DISK_BUDGET_MB": "200000",
    }
    assert spill_dir in worker_dirs(spec)
    assert not any(k.startswith("POLARS_OO") for k in scheduler_env(spec))


def test_env_vars_override_out_of_core_settings() -> None:
    spec = build_spec(
        license="/license.json",
        env_vars={"POLARS_OOMKILL_THRESHOLD_MB": "100", "POLARS_TEMP_DIR": "/t"},
    )

    env = worker_env(spec)
    assert env["POLARS_OOMKILL_THRESHOLD_MB"] == "100"
    assert env["POLARS_TEMP_DIR"] == "/t"


def test_the_eula_is_accepted_for_every_engine_by_default() -> None:
    spec = build_spec(license="/license.json")

    assert scheduler_env(spec)["POLARS_EULA_ACCEPTED"] == "yes"
    assert worker_env(spec)["POLARS_EULA_ACCEPTED"] == "yes"


def test_declining_the_eula_leaves_it_unset() -> None:
    spec = build_spec(license="/license.json", accept_eula=False)

    assert "POLARS_EULA_ACCEPTED" not in scheduler_env(spec)
    assert "POLARS_EULA_ACCEPTED" not in worker_env(spec)


def test_env_vars_can_accept_the_eula_too() -> None:
    spec = build_spec(
        license="/license.json",
        accept_eula=False,
        env_vars={"POLARS_EULA_ACCEPTED": "1"},
    )

    assert scheduler_env(spec)["POLARS_EULA_ACCEPTED"] == "1"


def test_overrides_merge_replace_and_delete() -> None:
    spec = build_spec(
        license="/license.json",
        scheduler_overrides={
            "scheduler": {
                "default_partitions_per_worker": 8,
                "checkpoint": {"period": "30s"},
                "client_service": {"bind_addr": "127.0.0.1:5051"},
                "deny_anonymous_users": None,
            },
            "observatory": {"database_path": "/data/observatory.db"},
            "monitoring": None,
            "memory_limit": 1024,
        },
        worker_overrides={"worker": {"heartbeat_period": "1s"}, "cpu_reserved": None},
    )

    scheduler = _scheduler(spec)
    assert scheduler["scheduler"]["default_partitions_per_worker"] == 8
    assert scheduler["scheduler"]["checkpoint"] == {"period": "30s"}
    assert scheduler["scheduler"]["client_service"] == {"bind_addr": "127.0.0.1:5051"}
    assert "deny_anonymous_users" not in scheduler["scheduler"]
    assert scheduler["scheduler"]["allow_local_scans"] is True
    assert scheduler["observatory"]["database_path"] == "/data/observatory.db"
    assert scheduler["observatory"]["enabled"] is True
    assert "monitoring" not in scheduler
    assert scheduler["memory_limit"] == 1024

    worker = _worker(spec)
    assert worker["worker"]["heartbeat_period"] == "1s"
    assert worker["worker"]["shuffle_location"] == {
        "local": {"path": "/tmp/polars/shuffle-data"}
    }
    assert "cpu_reserved" not in worker


def test_cluster_from_yaml_shaped_mappings() -> None:
    yaml_shaped: dict[str, typing.Any] = {
        "license": {
            "workspace_id": "ws",
            "client_id": "id",
            "client_secret": "secret",
            "cert_dir": "/certs",
        },
        "name": "from-yaml",
        "workers": 2,
        "shuffle_data": {"url": "s3://bucket/shuffle", "options": {"aws_region": "x"}},
        "anonymous_results": "gs://bucket/results",
        "ports": {"client": 6051, "task": 6000},
        "tls": {
            "cert_file": "/cert.pem",
            "key_file": "/key.pem",
            "auth": {"url": "https://idp/jwks", "issuer": "idp", "audience": "aud"},
        },
        "scheduler_overrides": {"scheduler": {"default_partitions_per_worker": 4}},
    }

    spec = build_spec(**yaml_shaped)

    scheduler = _scheduler(spec, ports=SCHEDULER_PORTS | {"client": 6051})
    assert scheduler["cluster_id"] == "from-yaml"
    assert scheduler["license"]["on_prem"]["cert_dir"] == "/certs"
    assert scheduler["scheduler"]["client_service"]["bind_addr"] == ":6051"
    assert scheduler["scheduler"]["client_service"]["connection"]["tls"] == {
        "certificate_key_file": "/cert.pem",
        "private_key_file": "/key.pem",
    }
    assert scheduler["scheduler"]["anonymous_result_location"] == {
        "gcs": {"url": "gs://bucket/results"}
    }
    assert scheduler["scheduler"]["default_partitions_per_worker"] == 4
    assert worker_ports(spec) == {"task": 6000, "shuffle": None}
    worker = _worker(spec)["worker"]
    assert worker["shuffle_location"] == {
        "s3": {"url": "s3://bucket/shuffle", "aws_region": "x"}
    }
