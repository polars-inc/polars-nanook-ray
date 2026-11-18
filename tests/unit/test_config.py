import pathlib

import pytest
import ray

from polars_nanook_ray import RayClusterContext


def test_construction_validates_without_ray() -> None:
    with pytest.raises(ValueError, match="workers"):
        RayClusterContext(license="/license.json", workers=-1)

    RayClusterContext(license="/license.json")
    assert not ray.is_initialized()


def test_relative_binary_path_made_absolute() -> None:
    ctx = RayClusterContext(license="/license.json", binary_path="bin/polars")

    assert ctx._spec.binary_path == str(pathlib.Path("bin/polars").absolute())
