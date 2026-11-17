import pathlib

import pydantic
import pytest
import toml

from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsLicenseServerConfig,
    PolarsRayClusterConfig,
    PolarsScalingConfig,
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
    "license_",
    [
        PolarsLicenseServerConfig(uri="https://license.example:50051"),
        {"license_type": "license_server", "uri": "https://license.example:50051"},
    ],
)
def test_license_server_uri_reaches_scheduler_and_workers(license_) -> None:
    config = _cluster_config(license=license_)

    scheduler = toml.loads(config.config_scheduler("10.0.0.1"))
    worker = toml.loads(
        config.config_worker(
            "worker-0", "10.0.0.2", "10.0.0.1", task_port=6000, shuffle_port=7000
        )
    )

    expected = {"license_server": {"uri": "https://license.example:50051"}}
    assert scheduler["license"] == expected
    assert worker["license"] == expected


def test_unpinned_scheduler_scales() -> None:
    scheduler = PolarsSchedulerConfig(
        placement_resource=None,
        scaling=PolarsScalingConfig(enabled=True, port=4001),
    )

    rendered = toml.loads(
        _cluster_config(scheduler=scheduler).config_scheduler("10.0.0.1")
    )

    assert rendered["scaling"]["rest"] == {"uri": "http://127.0.0.1:4001"}
