import logging
import os

from polars_nanook_ray.actors import (
    PolarsLicenseServerActor,
    PolarsScalerActor,
    PolarsSchedulerActor,
    PolarsWorkerActor,
)
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    PolarsCheckpointConfig,
    PolarsEnterpriseLicenseConfig,
    PolarsLicenseConfig,
    PolarsLicenseServerConfig,
    PolarsLineageConfig,
    PolarsMonitoringConfig,
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsRayLicenseServerConfig,
    PolarsScalingConfig,
    PolarsSchedulerConfig,
    PolarsServiceAccountLicenseConfig,
    PolarsSubprocessConfig,
    PolarsWorkerConfig,
)
from polars_nanook_ray.context import RayClusterContext
from polars_nanook_ray.license_server import PolarsRayLicenseServer

__all__ = [
    "PolarsCheckpointConfig",
    "PolarsEnterpriseLicenseConfig",
    "PolarsLicenseConfig",
    "PolarsLicenseServerActor",
    "PolarsLicenseServerConfig",
    "PolarsLineageConfig",
    "PolarsMonitoringConfig",
    "PolarsObservatoryConfig",
    "PolarsRayCluster",
    "PolarsRayClusterConfig",
    "PolarsRayLicenseServer",
    "PolarsRayLicenseServerConfig",
    "PolarsScalerActor",
    "PolarsScalingConfig",
    "PolarsSchedulerActor",
    "PolarsSchedulerConfig",
    "PolarsServiceAccountLicenseConfig",
    "PolarsSubprocessConfig",
    "PolarsWorkerActor",
    "PolarsWorkerConfig",
    "RayClusterContext",
]

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())
