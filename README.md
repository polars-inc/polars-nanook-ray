# Polars Nanook over Ray

Runs the Polars Nanook scheduler and workers as [Ray](https://www.ray.io/) actors.

## Quickstart

Prerequisites for a completely local test deployment:

- A Polars Nanook `license.json` file, or a valid service account
- A Python virtual environment:

```sh
uv venv
source .venv/bin/activate
uv pip install polars-nanook-ray
```

It will also install `ray`, and `polars-nanook-compute` which runs the scheduler
and workers, as dependencies. The latter pins the Polars version of the
environment, and is only installed on Linux, the only platform it is released
for: the Ray nodes running the scheduler and workers need to be Linux hosts.

Start a Ray head node:

```sh
ray start \
  --dashboard-host=0.0.0.0 \
  --disable-usage-stats \
  --head \
  --port="6379" \
  --ray-client-server-port="10001"
```

Then start a cluster of four workers on this host and run a query on it:

```py
import polars as pl
import ray

from polars_nanook_ray import RayComputeContext

ray.init(address="auto")

with RayComputeContext(
    license="/path/to/license.json",
    accept_eula=True,  # see the license section below
    workers=4,  # 1 CPU and 2 GiB each: Ray places them where they fit
) as ctx:
    print(
        pl.LazyFrame({"a": [1, 2, 3], "b": [4, 4, 5]})
        .with_columns(pl.col("a").max().over("b").alias("c"))
        .remote(ctx)
        .execute()
        .head
    )

ray.shutdown()
```

`RayComputeContext` takes keyword arguments only, and checks them all when it is
built, without touching Ray. A bad value raises a `ValueError` that names the
keyword. Entering the `with` block starts the cluster and waits until it is ready;
leaving it stops the cluster. `ctx.start()` and `ctx.stop()` do the same without a
`with` block.

`name` (default `"polars-nanook"`) is the cluster ID and the Ray namespace of its
actors. The package passes the namespace to every Ray call itself, so
`ray.init(address="auto")` needs none. Use a different `name` per cluster to run
several on one Ray cluster.

The package creates the local directories the engines need, such as
`/tmp/polars/...`, on each node before starting them.

`ctx.client_addr` is the URL queries are sent to, for example
`http://10.0.0.5:5051`. `ctx.dashboard_addr` is the URL of the observatory
dashboard, or `None` with `observatory=False`. Both raise while the cluster is not
running.

## License

`license` is required. It takes one of:

- a path to a license file, such as `"/etc/polars/license.json"`;
- the URI of a Polars Nanook license server, such as
  `"https://license-server.example.com:50051"`;
- a service account, to validate the license online:

```py
from polars_nanook_ray import ServiceAccount

license = ServiceAccount(
    workspace_id="<WORKSPACE_ID>",
    client_id="<SERVICE_ACCOUNT_ID>",
    client_secret="<SERVICE_ACCOUNT_SECRET>",
)
```

The secret does not show in reprs or logs. If you do not already have a service
account, create one through the
[Polars Nanook portal](https://nanook.pola.rs/api/redirects/register).

With a license server, the scheduler and the workers register with it. This
package does not run the license server: deploy it as its own service, outside
Ray, and start it before any cluster that points at it.

With a license file, the engine asks you to accept its EULA. Accept it with
`accept_eula=True`, which sets `POLARS_EULA_ACCEPTED=yes` for every engine. Setting
that variable on the Ray nodes, or through `env_vars`, works too. Without it the
engine prints the EULA and refuses to start.

## Storage

`shuffle_data` and `anonymous_results` each take a URL or an
absolute path. The scheme picks the backend:

- `s3://` or `s3a://`: S3
- `gs://` or `gcs://`: Google Cloud Storage
- `az://`, `azure://`, `abfs://` or `abfss://`: Azure Blob Storage
- no scheme: a path on the local disk of each node

```py
RayComputeContext(
    license="/path/to/license.json",
    shuffle_data="s3://bucket/shuffle",
    anonymous_results="s3://bucket/results",
)
```

Use `Storage` for credentials, other object store options, or an S3-compatible
endpoint such as MinIO:

```py
from polars_nanook_ray import Storage

shuffle_data = Storage(
    "s3://bucket/shuffle",
    options={"aws_region": "eu-west-1"},
    endpoint="http://minio:9000",
)
```

For a filesystem mounted at the same path on every node, such as NFS, use
`Storage("/mnt/nfs/shuffle", shared=True)`.

`shuffle_data` and `anonymous_results` default to `/tmp/polars/shuffle-data` and
`/tmp/polars/anonymous-results`. `spill_dir`
(default `/tmp/polars/temporary-data`) is where Polars spills to disk on each
worker.

## Ports and firewalls

Every port the engines listen on is set in one place, `ports`:

```py
from polars_nanook_ray import Ports

ports = Ports(
    client=5051,  # scheduler: query submission
    dashboard=3001,  # scheduler: observatory dashboard and REST API
    registration=5050,  # scheduler: workers register here
    telemetry=5049,  # scheduler: OpenTelemetry traces and metrics
    task=5052,  # each worker: tasks from the scheduler
    shuffle=5053,  # each worker: shuffle data from other workers
)
```

A port left at `None` is picked free on its node. By default only the client port
(5051) and the dashboard (3001) are fixed, as `polars_nanook` expects; all others
are free, so several workers and clusters can share a host. Ray spreads the
workers over the nodes as best it can, but may still put two on one node; with
fixed `task` or `shuffle` ports the second then fails to start, with a
`PortInUseError`. Leave those two unset unless every node has room for one worker
only.

`tls=Tls(cert_file=..., key_file=...)` serves the client port and the dashboard
over TLS. `Tls(..., auth=Jwks(url=..., issuer=..., audience=...))` also makes
both require a JWT validated against that JWKS endpoint:

```py
from polars_nanook_ray import Jwks, RayComputeContext, Tls

RayComputeContext(
    license="/path/to/license.json",
    tls=Tls(
        cert_file="/etc/polars/cert.pem",
        key_file="/etc/polars/key.pem",
        auth=Jwks(url="https://idp/jwks", issuer="https://idp", audience="polars"),
    ),
)
```

## Worker placement

Ray places the workers, as it does Spark executors in RayDP: each worker actor
requests `cpus` and `memory` and uses Ray's `SPREAD` scheduling strategy, without
pinning to a node. Ray puts it on a node with that much free, taking the nodes in
turn; this is best-effort, and a node with room for several may get several, and nodes without CPUs,
like a head started with `--num-cpus=0`, get none unless `cpus=0`.

A worker for which no node has room yet waits, as Ray demand: Ray's autoscaler, if
the Ray cluster has one, adds a node for it. It counts as one of the cluster's
workers meanwhile, and `start()` reports it if it is still waiting at
`startup_timeout`. Only a worker larger than every alive node fails `start()` right
away, naming the request and the largest node, and stopping what it started.

Workers are named by index, `worker-0`, `worker-1`, ..., each new one taking the
lowest free index; the name says nothing of the node. Starting a worker whose name
is taken returns the existing one, so repeated scale-ups and reconnects never add
workers: `start()` only tops the pool up to `workers`. Each start of a worker's
engine registers with the scheduler under a new instance ID, e.g.
`worker-0-1a2b3c4d`.

## Scheduler placement

The scheduler runs on the Ray head node by default, pinned with Ray's built-in
`node:__internal_head__` resource; no custom resource is needed. It requests no
CPUs, so a head node started with `--num-cpus=0` works too.

To pin it to another node, give that node a custom resource, for example
`ray start --resources='{"scheduler_node": 1}'`, and set
`scheduler_resource="scheduler_node"`. `scheduler_resource=None` lets Ray place
the scheduler on any node.

## Lifetime and reconnecting

The scheduler actor runs in `detached` mode and outlives the script that started
it. The workers belong to the scheduler actor: they stop when it does, and a
restarted scheduler starts new ones. `worker_max_restarts` and
`scheduler_max_restarts` (default `3` each) bound how often Ray restarts them; Ray
restarts a worker whose node is lost on another node. Every `reconcile_interval`
seconds (default `10`, `0` turns it off) the scheduler starts workers up to its
target count, replacing those past their restart limit.

Each actor waits up to `startup_timeout` seconds (default `30`) for its engine to
accept connections. An engine that exits first, or never listens, fails its actor
without Ray restarting it, as restarting does not fix a bad license or setting, and
`start()` raises with the errors the engine logged, after stopping the actors it
created:

```text
EngineStartError: The scheduler engine exited with code 1 while starting: License file check failed: Failed to read license file: No such file or directory (os error 2)
```

Ray restarts an actor whose engine dies after it started, up to its restart limit.

To attach to a running cluster from another script, connect by name:

```py
import ray

from polars_nanook_ray import RayComputeContext

ray.init(address="auto")
ctx = RayComputeContext.connect("polars-nanook")
print(ctx.client_addr)
ctx.stop()
```

`connect()` reads the cluster's settings back from its scheduler, so you do not
repeat them, and raises if no cluster of that name runs. Calling `start()` on a
context whose cluster already runs reconnects to it and only tops the workers up.

## Logging

The actors log to their Ray logs at INFO, or at the level the `LOGLEVEL`
environment variable sets. Importing the package leaves your own process's
logging alone: to see the cluster's progress messages in your script, set up
logging there, for example with `logging.basicConfig(level=logging.INFO)`.

`log_level` (`"info"`, `"debug"` or `"trace"`) sets the engines' own log level.
`env_vars` sets environment variables of every engine process, and so of their
query executors. `observatory=False` turns off the dashboard. Host metrics and
lineage events are set through the [engine overrides](#engine-overrides-and-yaml).

To clean all actors and underlying processes, Ray itself needs to be shut down
using the following command:

```sh
ray stop --force
```

## Autoscaling

With `autoscale=Autoscale(...)`, the scheduler actor runs an HTTP server to handle
scaling requests sent by the engine, and adds or removes Ray worker actors in
response. `Autoscale(min_workers=0, max_workers=None)` bounds how far it may scale,
`max_workers=None` leaving it unbounded; `autoscale=None`, the default, keeps
`workers` fixed.

The cluster always _starts_ `workers` workers: the bounds are reported back to the
engine on `GET /scale_config`, seeded into the per-query limits, and only enforced
on rescaling, where a `POST /scale_to` count outside them is clamped. A
scale-down that names no workers to keep or delete removes the highest indices
first. A scale-up starts workers whether or not a node has room for them yet, so
an autoscaling Ray cluster grows with the engine's requests. Requesting more
workers is done via the client: `.distributed(min_workers=X)`.

The HTTP server is unauthenticated, binds a free port on `127.0.0.1`, and is only
used by the engine. It accepts only `application/json` bodies, and ignores worker
names that are not worker actors.

> [!NOTE]
> A query that does not request a worker count is capped at
> `default_workers_per_query`, which this package sets to `workers`.
> `max_workers_per_query` bounds what any single query may claim, and is set to
> `Autoscale.max_workers`, or `workers` without it. To let queries trigger a scale-up,
> request workers per query via `.distributed(min_workers=X)`, or raise the
> default through `scheduler_overrides`:
>
> ```py
> scheduler_overrides={"scheduler": {"default_workers_per_query": 8}}
> ```

```py
from polars_nanook_ray import Autoscale, RayComputeContext

RayComputeContext(
    license="/path/to/license.json",
    workers=2,
    autoscale=Autoscale(min_workers=1, max_workers=16),
)
```

## Resource requests and limits

`cpus` (default `1`) and `memory` (default `2`, in GiB) are Ray's `num_cpus` and
`memory` actor options, used by Ray for bin-packing. `cpus` is also the engine's
CPU reservation, a hint for task placement reported to the observatory. Neither is
enforced at the OS level.

`worker_overrides={"memory_limit": <bytes>}` makes each worker engine also cap
itself through cgroups, but only if a _delegated cgroup subtree is made
available_. A plain session/SSH shell does not provide one (everything lives
flatly in one cgroup), so the memory limit fails outright with the following
message:

> Ensure cgroup is mounted and subgroups are delegated, or disable the memory
> limit in the configuration file.

`binary_path` runs a specific engine binary instead of the one from
`polars-nanook-compute`.

## Engine overrides and YAML

Engine settings without a keyword stay reachable through `scheduler_overrides`
and `worker_overrides`. These are merged into the engine's TOML configuration
last: mappings merge, `None` deletes a key, and any other value replaces it.

```py
RayComputeContext(
    license="/path/to/license.json",
    scheduler_overrides={"scheduler": {"default_partitions_per_worker": 4}},
    worker_overrides={"worker": {"heartbeat_period": "10s"}},
)
```

These settings have no keyword and are set this way:

| Setting | Override |
| --- | --- |
| Reject anonymous queries | `scheduler_overrides={"scheduler": {"deny_anonymous_users": True}}` |
| No scans or sinks on local disk | `scheduler_overrides={"scheduler": {"allow_local_scans": False, "allow_local_sinks": False}}` |
| No host metrics | `{"monitoring": {"host_metrics": {"enabled": False}}}` in both overrides |
| Lineage events to stdout | `{"lineage": {"transport": {"stdio": {"output_stream": "stdout"}}}}` in both overrides |
| Lineage events to an endpoint | `{"lineage": {"transport": {"http": {"endpoint": "https://..."}}}}` in both overrides |
| Cap worker memory | `worker_overrides={"memory_limit": <bytes>}`, see [resource limits](#resource-requests-and-limits) |

`license`, the storage keywords, `ports`, `autoscale` and `tls` also accept plain
mappings, so a whole cluster can come from a YAML file:

```py
import yaml

from polars_nanook_ray import RayComputeContext

with open("cluster.yaml") as f:
    ctx = RayComputeContext(**yaml.safe_load(f))
```

```yaml
license: /etc/polars/license.json
workers: 8
cpus: 15
memory: 56
ports:
  task: 5052
  shuffle: 5053
shuffle_data:
  url: s3://bucket/shuffle
  options:
    aws_region: eu-west-1
```
