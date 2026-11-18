import pathlib

import pytest
import ray

from polars_nanook_ray import RayComputeContext


def test_construction_validates_without_ray() -> None:
    with pytest.raises(ValueError, match="workers"):
        RayComputeContext(license="/license.json", workers=-1)

    RayComputeContext(license="/license.json")
    assert not ray.is_initialized()


def test_the_eula_is_accepted_by_default() -> None:
    ctx = RayComputeContext(license="/license.json")

    assert ctx._spec.accept_eula


def test_accept_eula_is_a_keyword() -> None:
    ctx = RayComputeContext(license="/license.json", accept_eula=False)

    assert not ctx._spec.accept_eula


def test_relative_binary_path_made_absolute() -> None:
    ctx = RayComputeContext(license="/license.json", binary_path="bin/polars")

    assert ctx._spec.binary_path == str(pathlib.Path("bin/polars").absolute())
