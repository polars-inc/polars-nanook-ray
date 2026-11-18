"""The parsed and validated keywords of a cluster, as shipped to its actors."""

import os
import pathlib
import typing
import urllib.parse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polars_nanook_ray.options import Autoscale, Ports, ServiceAccount, Storage, Tls

# advertised by every Ray head node, see `ray._private.ray_constants`
HEAD_NODE_RESOURCE = "node:__internal_head__"


class LicenseFile(BaseModel):
    """An offline license file, at the same absolute path on every node."""

    model_config = ConfigDict(frozen=True)

    path: str


class LicenseServer(BaseModel):
    """A license server the engines validate the license against."""

    model_config = ConfigDict(frozen=True)

    uri: str


def _absolute(path: str) -> str:
    return os.path.normpath(pathlib.Path(path).expanduser().absolute())


def _parse_license(value: typing.Any) -> typing.Any:
    if isinstance(value, ServiceAccount | LicenseFile | LicenseServer):
        return value
    if isinstance(value, typing.Mapping):
        return ServiceAccount.model_validate(value)
    if isinstance(value, str | os.PathLike):
        location = os.fspath(value)
        url = urllib.parse.urlparse(location)
        if url.scheme in ("http", "https"):
            return LicenseServer(uri=location)
        if url.scheme == "file":
            return LicenseFile(path=_absolute(url.netloc + url.path))
        if not url.scheme:
            return LicenseFile(path=_absolute(location))
        msg = f"unsupported license scheme {url.scheme!r}: use a file path or an http(s) URI"
        raise ValueError(msg)
    msg = "expected a file path, an http(s) URI, a ServiceAccount or a mapping"
    raise ValueError(msg)


def _parse_storage(value: typing.Any) -> typing.Any:
    if isinstance(value, str | os.PathLike):
        return Storage(value)
    return value


class ClusterSpec(BaseModel):
    """Everything a cluster is built from; see `RayClusterContext` for the keywords."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    license: LicenseFile | LicenseServer | ServiceAccount
    name: str = Field(default="polars-nanook", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    workers: int = Field(default=1, ge=0)
    cpus: float = Field(default=1, ge=0)
    memory: float = Field(default=2, ge=0)
    autoscale: Autoscale | None = None
    shuffle_data: Storage = Storage("/tmp/polars/shuffle-data")
    anonymous_results: Storage = Storage("/tmp/polars/anonymous-results")
    spill_dir: str = "/tmp/polars/temporary-data"
    ports: Ports = Ports()
    tls: Tls | None = None
    worker_max_restarts: int = Field(default=3, ge=-1)
    scheduler_max_restarts: int = Field(default=3, ge=-1)
    reconcile_interval: float = Field(default=10.0, ge=0)
    startup_timeout: float = Field(default=30.0, gt=0)
    scheduler_resource: str | None = HEAD_NODE_RESOURCE
    binary_path: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    log_level: typing.Literal["info", "debug", "trace"] = "info"
    observatory: bool = True
    scheduler_overrides: dict[str, typing.Any] = Field(default_factory=dict)
    worker_overrides: dict[str, typing.Any] = Field(default_factory=dict)

    @property
    def memory_bytes(self) -> int:
        """The memory of each worker, in bytes."""
        return int(self.memory * 1024**3)

    @field_validator("license", mode="before")
    @classmethod
    def _license(cls, value: typing.Any) -> typing.Any:
        return _parse_license(value)

    @field_validator("shuffle_data", "anonymous_results", mode="before")
    @classmethod
    def _storage(cls, value: typing.Any) -> typing.Any:
        return _parse_storage(value)

    @field_validator("spill_dir", mode="before")
    @classmethod
    def _spill_dir(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, str | os.PathLike):
            path = pathlib.Path(value).expanduser()
            if not path.is_absolute():
                msg = f"must be an absolute path, got {os.fspath(value)!r}"
                raise ValueError(msg)
            return str(path)
        return value

    @field_validator("binary_path", mode="before")
    @classmethod
    def _binary_path(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, str | os.PathLike):
            return _absolute(os.fspath(value))
        return value

    @field_validator("ports", mode="before")
    @classmethod
    def _ports(cls, value: typing.Any) -> typing.Any:
        return Ports() if value is None else value

    @field_validator(
        "env_vars", "scheduler_overrides", "worker_overrides", mode="before"
    )
    @classmethod
    def _empty(cls, value: typing.Any) -> typing.Any:
        return {} if value is None else value


StorageLike: typing.TypeAlias = (
    str | os.PathLike[str] | Storage | typing.Mapping[str, typing.Any]
)


def build_spec(
    *,
    license: str | os.PathLike[str] | ServiceAccount | typing.Mapping[str, typing.Any],
    name: str = "polars-nanook",
    workers: int = 1,
    cpus: float = 1,
    memory: float = 2,
    autoscale: Autoscale | typing.Mapping[str, int | None] | None = None,
    shuffle_data: StorageLike = "/tmp/polars/shuffle-data",
    anonymous_results: StorageLike = "/tmp/polars/anonymous-results",
    spill_dir: str | os.PathLike[str] = "/tmp/polars/temporary-data",
    ports: Ports | typing.Mapping[str, int | None] | None = None,
    tls: Tls | typing.Mapping[str, typing.Any] | None = None,
    worker_max_restarts: int = 3,
    scheduler_max_restarts: int = 3,
    reconcile_interval: float = 10.0,
    startup_timeout: float = 30.0,
    scheduler_resource: str | None = HEAD_NODE_RESOURCE,
    binary_path: str | os.PathLike[str] | None = None,
    env_vars: typing.Mapping[str, str] | None = None,
    log_level: typing.Literal["info", "debug", "trace"] = "info",
    observatory: bool = True,
    scheduler_overrides: typing.Mapping[str, typing.Any] | None = None,
    worker_overrides: typing.Mapping[str, typing.Any] | None = None,
) -> ClusterSpec:
    """Parse and validate the keywords of `RayClusterContext`.

    Raises `ValueError` naming the keyword at fault.
    """
    return ClusterSpec.model_validate(locals())
