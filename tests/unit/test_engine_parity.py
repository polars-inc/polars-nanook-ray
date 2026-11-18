"""The new keywords render what the old config classes rendered.

Goes away with `polars_nanook_ray.config`.
"""

import typing

import pytest
import toml

from polars_nanook_ray._engine import (
    SchedulerAddress,
    render_scheduler,
    render_worker,
    scheduler_env,
    worker_env,
)
from polars_nanook_ray._spec import build_spec
from polars_nanook_ray.config import (
    AbsLocationConfig,
    GcsLocationConfig,
    LineageHttpConfig,
    PolarsCheckpointConfig,
    PolarsEnterpriseLicenseConfig,
    PolarsJwksAuthConfig,
    PolarsLineageConfig,
    PolarsMonitoringConfig,
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsScalingConfig,
    PolarsSchedulerConfig,
    PolarsTlsConnectionConfig,
    PolarsWorkerConfig,
    S3LocationConfig,
)
from polars_nanook_ray.options import Jwks, Ports, Storage, Tls

if typing.TYPE_CHECKING:
    from polars_nanook_ray._spec import ClusterSpec

LICENSE = PolarsEnterpriseLicenseConfig(license_path="/license.json")


def _default() -> tuple[PolarsRayClusterConfig, "ClusterSpec"]:
    old = PolarsRayClusterConfig(
        num_workers=1,
        license=LICENSE,
        scheduler=PolarsSchedulerConfig(),
        worker=PolarsWorkerConfig(cpu_reserved=1, task_port=6000, shuffle_port=7000),
    )
    new = build_spec(license="/license.json", ports=Ports(task=6000, shuffle=7000))
    return old, new


def _integration() -> tuple[PolarsRayClusterConfig, "ClusterSpec"]:
    old = PolarsRayClusterConfig(
        cluster_id="polars-nanook-test",
        num_workers=2,
        min_workers=1,
        max_workers=4,
        license=LICENSE,
        scheduler=PolarsSchedulerConfig(
            cpus_hint=0,
            memory_hint=1,
            cpu_reserved=1,
            client_port=6051,
            worker_registration_port=6050,
            observatory=PolarsObservatoryConfig(enabled=False),
            scaling=PolarsScalingConfig(enabled=True, port=4100),
        ),
        worker=PolarsWorkerConfig(cpus_hint=0, memory_hint=1, cpu_reserved=1),
        monitoring=PolarsMonitoringConfig(enabled=False),
    )
    new = build_spec(
        license="/license.json",
        name="polars-nanook-test",
        workers=2,
        autoscale=True,
        min_workers=1,
        max_workers=4,
        cpus=1,
        observatory=False,
        host_metrics=False,
        ports={"client": 6051, "registration": 6050},
        scheduler_overrides={"cpu_reserved": "1.0"},
    )
    return old, new


def _secured() -> tuple[PolarsRayClusterConfig, "ClusterSpec"]:
    tls = PolarsTlsConnectionConfig(
        certificate_key_file="/certs/cert.pem", private_key_file="/certs/key.pem"
    )
    auth = PolarsJwksAuthConfig(
        url="https://idp.example/jwks", issuers="https://idp.example", audience="aud"
    )
    old = PolarsRayClusterConfig(
        num_workers=3,
        accept_eula=True,
        log_level="debug",
        license=LICENSE,
        scheduler=PolarsSchedulerConfig(
            connection=tls,
            auth=auth,
            deny_anonymous_users=True,
            allow_local_scans=False,
            allow_local_sinks=False,
            anonymous_result=GcsLocationConfig(
                url="gs://bucket/results", presign_duration="5 secs"
            ),
            checkpoint=PolarsCheckpointConfig(period="10s"),
            observatory=PolarsObservatoryConfig(rest_connection=tls, rest_auth=auth),
        ),
        worker=PolarsWorkerConfig(
            cpu_reserved=4,
            memory_limit=8 * 1024**3,
            task_port=6000,
            shuffle_port=7000,
            shuffle_data=S3LocationConfig(
                url="s3://bucket/shuffle",
                options={"aws_region": "eu-west-1"},
                external_endpoint="http://minio:9000",
                allow_deletes=True,
            ),
            checkpoint_location=AbsLocationConfig(
                url="az://container/checkpoints", options={"account_name": "acc"}
            ),
            env_vars={"FOO": "bar"},
            temporary_data_dir="/scratch",
        ),
        lineage=PolarsLineageConfig(
            http=LineageHttpConfig(endpoint="https://lineage.example")
        ),
    )
    new = build_spec(
        license="/license.json",
        accept_eula=True,
        log_level="debug",
        workers=3,
        cpus=4,
        memory=8,
        enforce_memory_limit=True,
        ports=Ports(task=6000, shuffle=7000),
        tls=Tls(cert_file="/certs/cert.pem", key_file="/certs/key.pem"),
        auth=Jwks(
            url="https://idp.example/jwks",
            issuer="https://idp.example",
            audience="aud",
        ),
        require_user=True,
        allow_local_io=False,
        anonymous_results=Storage("gs://bucket/results", presign_duration="5 secs"),
        shuffle_data=Storage(
            "s3://bucket/shuffle",
            options={"aws_region": "eu-west-1"},
            endpoint="http://minio:9000",
            allow_deletes=True,
        ),
        checkpoints=Storage(
            "az://container/checkpoints", options={"account_name": "acc"}
        ),
        env_vars={"FOO": "bar"},
        spill_dir="/scratch",
        lineage="https://lineage.example",
        scheduler_overrides={"scheduler": {"checkpoint": {"period": "10s"}}},
    )
    return old, new


CASES = [_default, _integration, _secured]


def _scheduler_ports(old: PolarsRayClusterConfig) -> dict[str, int]:
    observatory = old.scheduler.observatory
    ports = {
        "client": old.scheduler.client_port,
        "registration": old.scheduler.worker_registration_port,
    }
    if observatory.enabled:
        ports |= {
            "dashboard": observatory.rest_port,
            "telemetry": observatory.otlp_port,
        }
    return ports


@pytest.mark.parametrize("case", CASES)
def test_scheduler_toml_unchanged(case: typing.Callable) -> None:
    old, new = case()

    rendered = render_scheduler(
        new,
        host="10.0.0.1",
        ports=_scheduler_ports(old),
        scaler_port=old.scheduler.scaling.port,
    )

    assert toml.loads(rendered) == toml.loads(old.config_scheduler("10.0.0.1"))


@pytest.mark.parametrize("case", CASES)
def test_worker_toml_unchanged(case: typing.Callable) -> None:
    old, new = case()

    rendered = render_worker(
        new,
        instance_id="worker-0",
        host="10.0.0.2",
        task_port=6000,
        shuffle_port=7000,
        scheduler=SchedulerAddress(
            host="10.0.0.1",
            registration_port=old.scheduler.worker_registration_port,
            telemetry_port=old.scheduler.observatory.otlp_port,
        ),
    )

    expected = old.config_worker(
        "worker-0", "10.0.0.2", "10.0.0.1", task_port=6000, shuffle_port=7000
    )
    assert toml.loads(rendered) == toml.loads(expected)


@pytest.mark.parametrize("case", CASES)
def test_environment_unchanged(case: typing.Callable) -> None:
    old, new = case()

    assert scheduler_env(new) == old.engine_env() | old.worker.env_vars
    assert worker_env(new) == old.engine_env() | old.worker.env_vars | {
        "POLARS_TEMP_DIR": old.worker.temporary_data_dir
    }
