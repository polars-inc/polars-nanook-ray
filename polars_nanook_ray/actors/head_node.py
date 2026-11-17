import typing

import ray

# advertised by every Ray head node, see `ray._private.ray_constants`
HEAD_NODE_RESOURCE = "node:__internal_head__"


def head_node_options(
    name: str,
    *,
    num_cpus: float,
    memory: int | None = None,
) -> dict[str, typing.Any]:
    """Ray actor options pinning the actor `name` to the head node.

    Raises a `RuntimeError` right away if the head node can never hold the actor,
    rather than leaving it pending forever.
    """
    requested = {
        "CPU": num_cpus,
        "memory": memory or 0,
        HEAD_NODE_RESOURCE: 0.001,
    }

    head = next(
        (
            node
            for node in ray.nodes()
            if node["Alive"] and HEAD_NODE_RESOURCE in node["Resources"]
        ),
        None,
    )
    if head is None:
        msg = f"Cannot place the {name}: no alive Ray head node found."
        raise RuntimeError(msg)

    for resource, amount in requested.items():
        total = head["Resources"].get(resource, 0)
        if amount > total:
            msg = (
                f"Cannot place the {name} on the Ray head node: it requests "
                f"{_describe(resource, amount)}, but the head node has "
                f"{_describe(resource, total)}. Lower its `cpus_hint` / `memory_hint`."
            )
            raise RuntimeError(msg)

    return {
        "num_cpus": num_cpus,
        "memory": memory,
        "resources": {HEAD_NODE_RESOURCE: requested[HEAD_NODE_RESOURCE]},
    }


def _describe(resource: str, amount: float) -> str:
    if resource == "memory":
        return f"{int(amount)} bytes of memory"
    return f"{amount:g} {resource}"
