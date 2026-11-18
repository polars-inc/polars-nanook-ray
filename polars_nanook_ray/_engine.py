"""The engine's TOML layout and environment, rendered from a `ClusterSpec`.

The only module that knows how the engine is configured.
"""

import dataclasses
import pathlib
import typing

import toml

from polars_nanook_ray._spec import LicenseFile, LicenseServer
from polars_nanook_ray.identity import SCHEDULER
from polars_nanook_ray.options import ServiceAccount

if typing.TYPE_CHECKING:
    from polars_nanook_ray._spec import ClusterSpec
    from polars_nanook_ray.options import Storage

# the scheduler actor serves scaling requests here, and its engine calls it
SCALER_HOST = "127.0.0.1"

OBSERVATORY_DATABASE = "/tmp/polars/observatory/data.db"
_OBSERVATORY_MAX_METRICS_BYTES = 100 * 1024**2
# the engine's own default, where workers send telemetry if the scheduler has none
_DEFAULT_TELEMETRY_PORT = 5049
_HEARTBEAT_PERIOD = "5s"


@dataclasses.dataclass(frozen=True)
class SchedulerAddress:
    """Where the workers reach the scheduler: its host and the ports it bound."""

    host: str
    registration_port: int
    telemetry_port: int | None


def scheduler_ports(spec: "ClusterSpec") -> dict[str, int | None]:
    """Return the ports the scheduler binds, by name; `None` asks for a free one."""
    ports = {"client": spec.ports.client, "registration": spec.ports.registration}
    if spec.observatory:
        ports |= {"dashboard": spec.ports.dashboard, "telemetry": spec.ports.telemetry}
    return ports


def worker_ports(spec: "ClusterSpec") -> dict[str, int | None]:
    """Return the ports a worker binds, by name; `None` asks for a free one."""
    return {"task": spec.ports.task, "shuffle": spec.ports.shuffle}


def local_dirs(spec: "ClusterSpec") -> list[str]:
    """Return the local directories the engines need on each node."""
    dirs = [
        storage.url
        for storage in (spec.shuffle_data, spec.anonymous_results, spec.checkpoints)
        if storage is not None and storage.backend == "local"
    ]
    dirs.append(spec.spill_dir)
    if spec.observatory:
        dirs.append(str(pathlib.PurePath(OBSERVATORY_DATABASE).parent))
    if isinstance(spec.license, ServiceAccount):
        dirs.append(spec.license.cert_dir)
    return dirs


def scheduler_env(spec: "ClusterSpec") -> dict[str, str]:
    """Return the environment of the scheduler's engine process."""
    return {
        "POLARS_EULA_ACCEPTED": "yes" if spec.accept_eula else "no",
        "PLC_LOG_LEVEL": spec.log_level,
        "RUST_BACKTRACE": "full",
        **spec.env_vars,
    }


def worker_env(spec: "ClusterSpec") -> dict[str, str]:
    """Return the environment of a worker's engine process."""
    return scheduler_env(spec) | {"POLARS_TEMP_DIR": spec.spill_dir}


def render_scheduler(
    spec: "ClusterSpec",
    *,
    host: str,
    ports: typing.Mapping[str, int],
    scaler_port: int | None = None,
) -> str:
    """Render the scheduler's TOML.

    Parameters
    ----------
    spec
        The cluster.
    host
        The IP address of the scheduler's node.
    ports
        The ports the scheduler binds, by the names `scheduler_ports` gives.
    scaler_port
        The port on `SCALER_HOST` serving scaling requests; required to autoscale.
    """
    leader = SchedulerAddress(host, ports["registration"], ports.get("telemetry"))
    doc: dict = {
        "instance_id": SCHEDULER.instance_id,
        "cluster_id": spec.name,
        "cluster_mode": "ray",
        "license": _license(spec),
        "static_leader": _static_leader(leader),
        "scheduler": _scheduler(spec, ports),
        "observatory": _observatory(spec, ports),
        "monitoring": _monitoring(spec),
        "scaling": _scaling(spec, scaler_port),
    }
    if spec.lineage is not None:
        doc["lineage"] = _lineage(spec.lineage)
    return toml.dumps(_merge(doc, spec.scheduler_overrides))


def render_worker(
    spec: "ClusterSpec",
    *,
    instance_id: str,
    host: str,
    task_port: int,
    shuffle_port: int,
    scheduler: SchedulerAddress,
) -> str:
    """Render a worker's TOML.

    Parameters
    ----------
    spec
        The cluster.
    instance_id
        The engine instance ID of this start of the worker.
    host
        The IP address of the worker's node.
    task_port
        The port the worker receives tasks on.
    shuffle_port
        The port the worker serves shuffle data on.
    scheduler
        Where the worker reaches the scheduler.
    """
    worker: dict = {
        "enabled": True,
        "task_service": {
            "bind_addr": f":{task_port}",
            "public_addr": f"{host}:{task_port}",
        },
        "shuffle_service": {
            "bind_addr": f":{shuffle_port}",
            "public_addr": f"{host}:{shuffle_port}",
        },
        "heartbeat_period": _HEARTBEAT_PERIOD,
        "shuffle_location": _storage(spec.shuffle_data),
    }
    if spec.checkpoints is not None:
        worker["checkpoint_location"] = _storage(spec.checkpoints)
    if spec.env_vars:
        worker["env_vars"] = dict(spec.env_vars)

    doc: dict = {
        "instance_id": instance_id,
        "cluster_id": spec.name,
        "cluster_mode": "ray",
        "static_leader": _static_leader(scheduler),
        "worker": worker,
        "monitoring": _monitoring(spec),
    }
    if spec.cpus:
        doc["cpu_reserved"] = str(spec.cpus)
    if spec.enforce_memory_limit:
        doc["memory_limit"] = spec.memory_bytes
    if not isinstance(spec.license, ServiceAccount):
        doc["license"] = _license(spec)
    if spec.lineage is not None:
        doc["lineage"] = _lineage(spec.lineage)
    return toml.dumps(_merge(doc, spec.worker_overrides))


def _license(spec: "ClusterSpec") -> dict:
    license_ = spec.license
    if isinstance(license_, LicenseFile):
        return {"on_prem_enterprise": {"license_path": license_.path}}
    if isinstance(license_, LicenseServer):
        return {"license_server": {"uri": license_.uri}}
    return {
        "on_prem": {
            "workspace_id": license_.workspace_id,
            "client_id": license_.client_id,
            "client_secret": license_.client_secret.get_secret_value(),
            "cert_dir": license_.cert_dir,
        }
    }


def _static_leader(leader: SchedulerAddress) -> dict:
    telemetry = leader.telemetry_port or _DEFAULT_TELEMETRY_PORT
    return {
        "leader_instance_id": SCHEDULER.instance_id,
        "scheduler_service": {
            "public_addr": f"{leader.host}:{leader.registration_port}"
        },
        "observatory_service": {"public_addr": f"{leader.host}:{telemetry}"},
    }


def _secured(spec: "ClusterSpec", bind_port: int) -> dict:
    service: dict = {"bind_addr": f":{bind_port}"}
    if spec.tls is not None:
        service["connection"] = {
            "tls": {
                "certificate_key_file": spec.tls.cert_file,
                "private_key_file": spec.tls.key_file,
            }
        }
    if spec.auth is not None:
        service["auth"] = {
            "jwks": {
                "url": spec.auth.url,
                "issuers": spec.auth.issuer,
                "audience": spec.auth.audience,
            }
        }
    return service


def _scheduler(spec: "ClusterSpec", ports: typing.Mapping[str, int]) -> dict:
    d: dict = {
        "enabled": True,
        "allow_local_scans": spec.allow_local_io,
        "allow_local_sinks": spec.allow_local_io,
        "deny_anonymous_users": spec.require_user,
        "client_service": _secured(spec, ports["client"]),
        "worker_service": {"bind_addr": f":{ports['registration']}"},
        "anonymous_result_location": _storage(spec.anonymous_results),
    }
    max_per_query = spec.max_workers or spec.workers
    default_per_query = spec.workers or spec.min_workers or int(spec.autoscale)
    if max_per_query:
        d["max_workers_per_query"] = max_per_query
    if default_per_query:
        d["default_workers_per_query"] = default_per_query
    return d


def _observatory(spec: "ClusterSpec", ports: typing.Mapping[str, int]) -> dict:
    if not spec.observatory:
        return {"enabled": False}
    return {
        "enabled": True,
        "max_metrics_bytes_total": _OBSERVATORY_MAX_METRICS_BYTES,
        "database_path": OBSERVATORY_DATABASE,
        "service": {"bind_addr": f":{ports['telemetry']}"},
        "rest_api": {"service": _secured(spec, ports["dashboard"])},
    }


def _monitoring(spec: "ClusterSpec") -> dict:
    if not spec.host_metrics:
        return {"enabled": False}
    return {"enabled": True, "host_metrics": {"enabled": True}}


def _scaling(spec: "ClusterSpec", scaler_port: int | None) -> dict:
    if not spec.autoscale:
        return {"enabled": False}
    if scaler_port is None:
        msg = "autoscaling needs the port serving scaling requests"
        raise ValueError(msg)
    return {"enabled": True, "rest": {"uri": f"http://{SCALER_HOST}:{scaler_port}"}}


def _storage(storage: "Storage") -> dict:
    match storage.backend:
        case "local":
            return {"local": {"path": storage.url}}
        case "shared":
            return {"shared_filesystem": {"path": storage.url}}
        case backend:
            section: dict = {"url": storage.url}
            if storage.endpoint is not None:
                section["external_endpoint"] = storage.endpoint
            if storage.presign_duration is not None:
                section["presign_duration"] = storage.presign_duration
            if storage.allow_deletes is not None:
                section["allow_deletes"] = storage.allow_deletes
            return {backend: section | storage.options}


def _lineage(lineage: str) -> dict:
    if lineage in ("stdout", "stderr"):
        return {"transport": {"stdio": {"output_stream": lineage}}}
    return {"transport": {"http": {"endpoint": lineage}}}


def _merge(base: dict, overrides: typing.Mapping[str, typing.Any]) -> dict:
    """Return `base` with `overrides` merged in: `None` deletes, mappings merge."""
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, typing.Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
