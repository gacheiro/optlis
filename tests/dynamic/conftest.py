import pytest

from optlis.dynamic.utils import load_instance


@pytest.fixture()
def example_dynamic_instance():
    return load_instance("data/instances/example-dynamic.dat")
