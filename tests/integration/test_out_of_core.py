import pathlib

import polars as pl

from .conftest import ContextFactory


def test_workers_spill_to_the_spill_dir(
    ray_cluster: ContextFactory, tmp_path: pathlib.Path
) -> None:
    spill_dir = tmp_path / "spill"
    ctx = ray_cluster(
        workers=1,
        out_of_core={"spill_dir": spill_dir, "memory_budget_mb": 1},
    )

    n = 2_000_000
    result = (
        pl.LazyFrame({"a": range(n)})
        .sort("a", descending=True)
        .remote(ctx)
        .execute()
        .head
    )

    assert result is not None
    # the engine spills under a directory named after its PID, created on its first
    # spill
    assert any(path.name.isdigit() for path in spill_dir.iterdir())
