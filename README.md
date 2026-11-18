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
Note that a few things need to be set up for the Polars Nanook cluster to
function properly:

```sh
# need to match the config if manually changed
mkdir --parents /tmp/polars/{anonymous-results,license,observatory,shuffle-data,temporary-data}
```

This latter can be started using the following command:

```sh
ray start \
  --dashboard-host=0.0.0.0 \
  --disable-usage-stats \
  --head \
  --port="6379" \
  --ray-client-server-port="10001"
```

The scheduler and the scaler run on the Ray head node by default,
pinned with Ray's built-in `node:__internal_head__` resource; no custom resource is
needed. They request no CPUs by default, so a head node started with `--num-cpus=0`
works too.

To pin them to another node, give that node a custom resource, for example
`ray start --resources='{"scheduler_node": 1}'`, and set
`PolarsSchedulerConfig(placement_resource="scheduler_node")`. With scaling enabled,
only one node may advertise it, as the scaler must share the scheduler's node.
`placement_resource=None` lets Ray place the scheduler on any node that fits its
`cpus_hint` and `memory_hint`; scaling then can't be enabled.

If you do not already have one, create a Polars Nanook service account through the
[Polars Nanook portal](https://nanook.pola.rs/api/redirects/register).

Spawn a local multinode cluster:

```py
import polars as pl
import polars_nanook as pn
import ray

from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsServiceAccountLicenseConfig,
    PolarsWorkerConfig,
)

config = PolarsRayClusterConfig(
    num_workers=4,
    license=PolarsServiceAccountLicenseConfig(
        workspace_id="<WORKSPACE_ID>",
        client_id="<SERVICE_ACCOUNT_ID>",
        client_secret="<SERVICE_ACCOUNT_SECRET>",
    ),
    scheduler=PolarsSchedulerConfig(...),
    worker=PolarsWorkerConfig(...),
)

ray.init(address="auto", namespace=config.cluster_id)
cluster = PolarsRayCluster(config)
cluster.start()

print(
    pl.LazyFrame({"a": [1, 2, 3], "b": [4, 4, 5]})
    .with_columns(pl.col("a").max().over("b").alias("c"))
    .remote(pn.ClusterContext(uri=f"http://{cluster.get_client_addr()}"))
    .execute()
    .head
)

cluster.stop()
ray.shutdown()
```

or:

```py
import polars as pl
import ray

from polars_nanook_ray.config import (
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsServiceAccountLicenseConfig,
    PolarsWorkerConfig,
)
from polars_nanook_ray.context import RayClusterContext

config = PolarsRayClusterConfig(
    num_workers=4,
    license=PolarsServiceAccountLicenseConfig(
        workspace_id="<WORKSPACE_ID>",
        client_id="<SERVICE_ACCOUNT_ID>",
        client_secret="<SERVICE_ACCOUNT_SECRET>",
    ),
    scheduler=PolarsSchedulerConfig(...),
    worker=PolarsWorkerConfig(...),
)

ray.init(address="auto", namespace=config.cluster_id)

with RayClusterContext(config) as ctx:
    print(
        pl.LazyFrame({"a": [1, 2, 3], "b": [4, 4, 5]})
        .with_columns(pl.col("a").max().over("b").alias("c"))
        .remote(ctx)
        .execute()
        .head
    )

ray.shutdown()
```

Each worker picks free ports on its node, so several workers can share a host.
To fix them instead, for firewall rules, set `task_port` and `shuffle_port` on
`PolarsWorkerConfig`. Ray spreads the workers over the nodes as best it can, but
may still put two on one node; with fixed ports the second then fails to start,
with a `PortInUseError`. Leave them unset unless every node has room for one
worker only.

## Worker placement

Ray places the workers, as it does Spark executors in RayDP: each worker actor
requests `cpus_hint` and `memory_hint` and uses Ray's `SPREAD` scheduling
strategy, without pinning to a node. Ray puts it on a node with that much free,
taking the nodes in turn; this is best-effort, and a node with room for several
may get several, and nodes without CPUs, like a head started with
`--num-cpus=0`, get none unless `cpus_hint=0`. Nodes are listed with
`ray.nodes()`, so neither `start()` nor `stop()` needs the Ray dashboard.

A worker for which no node has room yet waits, as Ray demand: Ray's autoscaler, if
the Ray cluster has one, adds a node for it. It counts as one of the cluster's
workers meanwhile, and `start()` reports it if it is still waiting at
`startup_timeout`. Only a worker larger than every alive node fails `start()` right
away, naming the request and the largest node.

Workers are named by index, `worker-0`, `worker-1`, ..., each new one taking the
lowest free index; the name says nothing of the node. Starting a worker whose name
is taken returns the existing one, so repeated scale-ups and reconnects never add
workers: `start()` only tops the pool up to `num_workers`. Each start of a worker's
engine registers with the scheduler under a new instance ID, e.g.
`worker-0-1a2b3c4d`.

The actors are running in `detached` mode and survive past the script: one can
reconnect with the same `ray.init()` and `cluster` gymnastics from another
process.
To clean all actors and underlying processes, Ray itself needs to be shutdown
using the following command:

```sh
ray stop --force
```

## Autoscaling

A dedicated scaler actor, pinned to the scheduler node, runs an HTTP server to
handle scaling requests sent by the underlying binary.
These requests are relayed to the scheduler actor, which in turn adds or removes
Ray worker actors in response.

Enable it via `PolarsScalingConfig` on the scheduler, and optionally set
`min_workers` and/or `max_workers` on the cluster config to bound how far it may
scale.
Note the cluster always _starts_ `num_workers` workers: the bounds are reported
back to the binary on `GET /scale_config`, seeded into `max_workers_per_query`,
and only enforced on rescaling, where a `POST /scale_to` count outside them is
clamped.
Requesting more workers is done via the client: `.distributed(min_workers=X)`.

The HTTP server is unauthenticated and binds `127.0.0.1`. It accepts only
`application/json` bodies, and ignores worker names that are not worker actors.

> [!NOTE]
> Avoid leaving `default_workers_per_query` at `1` if you want queries to ever
> trigger a scale-up.
>
> A query that does not request a worker count is capped at
> `default_workers_per_query`, which this package seeds from `num_workers`, then
> `min_workers`, then `1` when autoscaling is enabled. Set it explicitly on
> `PolarsSchedulerConfig`, or request workers per query via
> `.distributed(min_workers=X)`, to scale past it. `max_workers_per_query` bounds
> what any single query may claim, and is seeded from `max_workers`, then
> `num_workers`.

## License server

To validate the license against a Polars Nanook license server, point the
cluster at it:

```py
from polars_nanook_ray.config import PolarsLicenseServerConfig

license = PolarsLicenseServerConfig(uri="https://license-server.example.com:50051")
```

The scheduler and the workers register with it. This package does not run the
license server: deploy it as its own service, outside Ray, and start it before
any cluster that points at it.

## Resource requests and limits

Four parameters control resource usage:

- `cpus_hint` / `memory_hint`: forwarded verbatim as Ray's own `num_cpus` /
  `memory` actor options, and used by Ray for bin-packing. Not enforced at the
  OS level.
- `cpu_reserved`: a scheduling/accounting hint consumed internally by the binary
  for task placement and reported to the observatory; never enforced.
- `memory_limit`: enforced by the binary itself via cgroups; but only if a
  _delegated cgroup subtree is made available_.

A plain session/SSH shell does not provide one (everything lives flatly in one
cgroup), so `memory_limit` fails outright with the following message:

> Ensure cgroup is mounted and subgroups are delegated, or disable the memory
> limit in the configuration file.
