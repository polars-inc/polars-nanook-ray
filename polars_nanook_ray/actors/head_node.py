import typing

import ray

from polars_nanook_ray._spec import HEAD_NODE_RESOURCE

# the scheduler runs on the head node by default, which often has no CPUs to give
SCHEDULER_CPUS = 0
SCHEDULER_MEMORY = 1024**3


def scheduler_options(resource: str | None) -> dict[str, typing.Any]:
    """Ray actor options placing the scheduler on the nodes advertising `resource`."""
    return placement_options(
        "scheduler", resource=resource, num_cpus=SCHEDULER_CPUS, memory=SCHEDULER_MEMORY
    )


def placement_options(
    name: str,
    *,
    resource: str | None,
    num_cpus: float,
    memory: int | None = None,
) -> dict[str, typing.Any]:
    """Ray actor options pinning the actor `name` to the nodes advertising `resource`.

    `None` leaves the placement to Ray. Raises a `RuntimeError` right away if no
    candidate node can ever hold the actor, rather than leaving it pending forever.
    """
    requested = {"CPU": num_cpus, "memory": memory or 0}
    where = _describe_placement(resource)

    candidates = [
        node
        for node in ray.nodes()
        if node["Alive"] and (resource is None or resource in node["Resources"])
    ]
    if not candidates:
        missing = (
            "found" if resource is None else f"advertises the `{resource}` resource"
        )
        msg = f"Cannot place the {name}: no alive Ray node {missing}."
        raise RuntimeError(msg)

    if not any(_fits(node, requested) for node in candidates):
        holder = "that node" if len(candidates) == 1 else "the largest one"
        for res, amount in requested.items():
            most = max(node["Resources"].get(res, 0) for node in candidates)
            if amount > most:
                shortfall = (
                    f"{_describe(res, amount)}, but {holder} has {_describe(res, most)}"
                )
                break
        else:
            shortfall = (
                f"{_describe('CPU', num_cpus)} and {_describe('memory', memory or 0)}, "
                "but no single node has both"
            )
        msg = (
            f"Cannot place the {name} on {where}: it requests {shortfall}. "
            "Free some on that node, or place it elsewhere with `scheduler_resource`."
        )
        raise RuntimeError(msg)

    options: dict[str, typing.Any] = {"num_cpus": num_cpus, "memory": memory}
    if resource is not None:
        options["resources"] = {resource: 0.001}
    return options


def _fits(node: dict[str, typing.Any], requested: dict[str, float]) -> bool:
    return all(
        node["Resources"].get(res, 0) >= amount for res, amount in requested.items()
    )


def _describe_placement(resource: str | None) -> str:
    if resource is None:
        return "any Ray node"
    if resource == HEAD_NODE_RESOURCE:
        return f"the Ray head node (`{resource}`)"
    return f"a Ray node with the `{resource}` resource"


def _describe(resource: str, amount: float) -> str:
    if resource == "memory":
        return f"{int(amount)} bytes of memory"
    return f"{amount:g} {resource}"
