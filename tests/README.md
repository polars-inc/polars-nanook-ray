# Local Ray cluster in Docker

[`docker-compose.yaml`](docker-compose.yaml) starts a Ray cluster of Linux
containers, to try the released package where the engine does not run natively,
such as macOS. It has one Ray head node, without CPUs, and Ray worker nodes, on
which Ray places the Polars Nanook workers. Every node installs
`polars-nanook-ray` from PyPI.

The commands below run from this directory.

## Start and stop the cluster

```sh
docker compose up --build --detach                 # 2 Ray worker nodes
docker compose up --detach --scale ray-worker=4    # or 4
docker compose down
```

The Ray dashboard is then at [http://localhost:8265](http://localhost:8265),
and the Polars Nanook observatory at
[http://localhost:3001](http://localhost:3001), as the scheduler runs on the
head node. These ports must be free on the host: stop any Ray running natively
first.

## Mount the license, scripts and data

The containers only see what is mounted in them. Uncomment the `volumes` section
of `docker-compose.yaml`, under `x-ray-node` so that every node mounts the same
files at the same paths, and edit it:

```yaml
x-ray-node: &ray-node
  ...
  volumes:
    - /path/to/license.json:/etc/polars/license.json:ro
    - /path/to/scripts:/usr/src
```

A script then passes `license="/etc/polars/license.json"`. Data a query reads or
writes on local disk must be mounted the same way, as the workers run on the
worker nodes. With a service account instead of a license file, there is no file
to mount.

If a file to mount does not exist, Docker mounts an empty directory in its place,
and creates it on the host too: the engine then fails to read the license.

## Run a script

Scripts run on the head node, where `ray.init(address="auto")` finds the cluster.
A script in a mounted directory:

```sh
docker compose exec ray-head python /usr/src/script.py
```

A script anywhere on the host, passed through standard input, needs no mount:

```sh
docker compose exec --no-tty ray-head python - < ~/path/to/script.py
```

The [quickstart example](../README.md#quickstart) runs either way, with the
license path above.
