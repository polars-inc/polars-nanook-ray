"""Settings of a cluster that group naturally, given to `RayClusterContext`."""

import os
import typing
import urllib.parse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


class ServiceAccount(BaseModel):
    """Validate the license online, through the Polars control plane.

    Parameters
    ----------
    workspace_id
        Control-plane workspace ID.
    client_id
        Control-plane OAuth client ID.
    client_secret
        Control-plane OAuth client secret, hidden from reprs and logs.
    cert_dir
        Directory on the scheduler's node for the TLS certificates of control-plane
        communication, created if missing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    client_id: str
    client_secret: SecretStr
    cert_dir: str = "/tmp/polars/license"

    def __init__(
        self,
        *,
        workspace_id: str,
        client_id: str,
        client_secret: str | SecretStr,
        cert_dir: str | os.PathLike[str] = "/tmp/polars/license",
    ) -> None:
        super().__init__(
            workspace_id=workspace_id,
            client_id=client_id,
            client_secret=client_secret,
            cert_dir=os.fspath(cert_dir),
        )


_OBJECT_STORES = {
    "s3": "s3",
    "s3a": "s3",
    "gs": "gcs",
    "gcs": "gcs",
    "az": "abs",
    "azure": "abs",
    "abfs": "abs",
    "abfss": "abs",
}


class Storage(BaseModel):
    """A storage location: an object store URL or an absolute path.

    The URL scheme picks the backend: `s3://` or `s3a://` for S3, `gs://` or `gcs://`
    for Google Cloud Storage, `az://`, `azure://`, `abfs://` or `abfss://` for Azure
    Blob Storage. A path without a scheme is on the local disk of each node.

    Parameters
    ----------
    url
        The location, e.g. `s3://bucket/prefix` or `/mnt/data`.
    options
        Object store options, such as credentials or a region. These are the keys of
        Object Store's `AmazonS3ConfigKey`, `GoogleConfigKey` or `AzureConfigKey`.
    endpoint
        Endpoint of an S3-compatible store, such as MinIO. S3 only.
    presign_duration
        How long presigned URLs stay valid, as an ISO 8601 (`PT5S`) or friendly
        (`5 secs`) duration. Object stores only.
    allow_deletes
        Whether the object store client may delete objects. Object stores only.
    shared
        Whether the path is a filesystem mounted at the same path on every node, such
        as NFS. Paths only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    options: dict[str, str] = Field(default_factory=dict)
    endpoint: str | None = None
    presign_duration: str | None = None
    allow_deletes: bool | None = None
    shared: bool = False

    def __init__(
        self,
        url: str | os.PathLike[str],
        /,
        *,
        options: typing.Mapping[str, str] | None = None,
        endpoint: str | None = None,
        presign_duration: str | None = None,
        allow_deletes: bool | None = None,
        shared: bool = False,
    ) -> None:
        super().__init__(
            url=url,
            options=options or {},
            endpoint=endpoint,
            presign_duration=presign_duration,
            allow_deletes=allow_deletes,
            shared=shared,
        )

    @field_validator("url", mode="before")
    @classmethod
    def _url(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str) and not urllib.parse.urlparse(value).scheme:
            value = os.path.expanduser(value)  # noqa: PTH111
            if not os.path.isabs(value):  # noqa: PTH117
                msg = f"a local path must be absolute, got {value!r}"
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _fits_backend(self) -> "Storage":
        scheme = urllib.parse.urlparse(self.url).scheme
        if scheme and scheme not in _OBJECT_STORES:
            known = ", ".join(f"{s}://" for s in _OBJECT_STORES)
            msg = f"unknown scheme {scheme!r} in {self.url!r}, expected one of {known} or an absolute path"
            raise ValueError(msg)
        if self.endpoint is not None and self.backend != "s3":
            msg = "`endpoint` applies to S3 only"
            raise ValueError(msg)
        if self.shared and scheme:
            msg = "`shared` applies to paths only"
            raise ValueError(msg)
        if not scheme and (
            self.options
            or self.presign_duration is not None
            or self.allow_deletes is not None
        ):
            msg = "`options`, `presign_duration` and `allow_deletes` apply to object stores only"
            raise ValueError(msg)
        return self

    @property
    def backend(self) -> typing.Literal["s3", "gcs", "abs", "local", "shared"]:
        """The kind of storage the URL names."""
        scheme = urllib.parse.urlparse(self.url).scheme
        if scheme:
            return _OBJECT_STORES[scheme]  # type: ignore[return-value]
        return "shared" if self.shared else "local"


class Ports(BaseModel):
    """The ports the engines listen on; `None` picks a free one on each node.

    Parameters
    ----------
    client
        The scheduler's port for query submission.
    dashboard
        The scheduler's port for the observatory dashboard and REST API.
    task
        Each worker's port for tasks from the scheduler.
    shuffle
        Each worker's port for shuffle data from other workers.
    registration
        The scheduler's port for workers to register on.
    telemetry
        The scheduler's port for OpenTelemetry traces and metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client: int | None = 5051
    dashboard: int | None = 3001
    task: int | None = None
    shuffle: int | None = None
    registration: int | None = None
    telemetry: int | None = None


class Tls(BaseModel):
    """Serve the client port and the dashboard over TLS.

    Parameters
    ----------
    cert_file
        Path to the PEM certificate on the scheduler's node.
    key_file
        Path to the PEM private key on the scheduler's node.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cert_file: str
    key_file: str


class Jwks(BaseModel):
    """Require a JWT validated against a JWKS endpoint on the client port and dashboard.

    Parameters
    ----------
    url
        The JWKS endpoint.
    issuer
        The accepted JWT issuer.
    audience
        The accepted JWT audience.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    issuer: str
    audience: str
