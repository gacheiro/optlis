import pytest

from optlis.dynamic.problem_data import load_instance


@pytest.fixture()
def example_dynamic_instance():
    return load_instance("data/instances/example-dynamic.dat")
