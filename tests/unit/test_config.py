import pathlib

import pytest
import ray

import polars_nanook_ray
from polars_nanook_ray import RayClusterContext


def test_only_the_public_names_are_exported() -> None:
    assert sorted(polars_nanook_ray.__all__) == [
        "Jwks",
        "Ports",
        "RayClusterContext",
        "ServiceAccount",
        "Storage",
        "Tls",
    ]


def test_construction_validates_without_ray() -> None:
    with pytest.raises(ValueError, match="workers"):
        RayClusterContext(license="/license.json", workers=-1)

    RayClusterContext(license="/license.json")
    assert not ray.is_initialized()


def test_relative_binary_path_made_absolute() -> None:
    ctx = RayClusterContext(license="/license.json", binary_path="bin/polars")

    assert ctx._spec.binary_path == str(pathlib.Path("bin/polars").absolute())


def test_not_running_until_started() -> None:
    ctx = RayClusterContext(license="/license.json")

    with pytest.raises(RuntimeError, match="not running"):
        ctx.client_addr
