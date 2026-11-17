import pydantic
import pytest
import toml

from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsSubprocessConfig,
    PolarsWorkerConfig,
)


def _cluster_config(**kwargs) -> PolarsRayClusterConfig:
    return PolarsRayClusterConfig.model_validate(
        {
            "license": PolarsEnterpriseLicenseConfig(license_path="/license.json"),
            "scheduler": PolarsSchedulerConfig(),
            "worker": PolarsWorkerConfig(task_port=6000, shuffle_port=7000),
            **kwargs,
        }
    )


def test_subprocess_runtime_is_the_default() -> None:
    assert _cluster_config().runtime == PolarsSubprocessConfig()


def test_subprocess_runtime_from_dict() -> None:
    config = _cluster_config(runtime={"runtime_type": "subprocess"})
    assert config.runtime == PolarsSubprocessConfig()


def test_docker_runtime_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="subprocess"):
        _cluster_config(runtime={"runtime_type": "docker", "image": "polars:latest"})


@pytest.mark.parametrize(
    ("single_host_cluster", "expected_task_port", "expected_shuffle_port"),
    [(False, 6000, 7000), (True, 6004, 7004)],
)
def test_worker_ports_offset_on_single_host(
    single_host_cluster: bool,
    expected_task_port: int,
    expected_shuffle_port: int,
) -> None:
    config = _cluster_config(
        runtime=PolarsSubprocessConfig(single_host_cluster=single_host_cluster),
    )

    worker = toml.loads(config.config_worker(2, "10.0.0.2", "10.0.0.1"))["worker"]

    assert worker["task_service"]["public_addr"] == f"10.0.0.2:{expected_task_port}"
    assert (
        worker["shuffle_service"]["public_addr"] == f"10.0.0.2:{expected_shuffle_port}"
    )
