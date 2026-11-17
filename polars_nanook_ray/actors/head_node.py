import typing

import ray

from polars_nanook_ray.config import HEAD_NODE_RESOURCE


def placement_options(
    name: str,
    *,
    resource: str | None,
    num_cpus: float,
    memory: int | None = None,
    single_node: bool = False,
) -> dict[str, typing.Any]:
    """Ray actor options pinning the actor `name` to the nodes advertising `resource`.

    `None` leaves the placement to Ray. Raises a `RuntimeError` right away if no
    candidate node can ever hold the actor, rather than leaving it pending forever,
    or, with `single_node`, if more than one node is a candidate.
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

    if single_node and len(candidates) > 1:
        msg = (
            f"Cannot place the {name} on {where}: {len(candidates)} alive Ray nodes "
            "qualify, but the scaler must share the scheduler's node. Advertise "
            "`placement_resource` on a single node."
        )
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
            "Lower its `cpus_hint` / `memory_hint`."
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
