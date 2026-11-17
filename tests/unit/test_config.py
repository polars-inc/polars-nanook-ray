import pathlib

import pydantic
import pytest
import toml

from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
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


def test_removed_runtime_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="`runtime` was removed"):
        _cluster_config(runtime={"binary_path": "/path/to/binary"})


def test_relative_binary_path_made_absolute() -> None:
    assert _cluster_config(binary_path="bin/polars").binary_path == str(
        pathlib.Path("bin/polars").absolute()
    )


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
        single_host_cluster=single_host_cluster,
    )

    worker = toml.loads(config.config_worker(2, "10.0.0.2", "10.0.0.1"))["worker"]

    assert worker["task_service"]["public_addr"] == f"10.0.0.2:{expected_task_port}"
    assert (
        worker["shuffle_service"]["public_addr"] == f"10.0.0.2:{expected_shuffle_port}"
    )
