import pytest

from instances import load_instance, import_solution
from instances.notebook_utils import (makespan, weighted_sum_completion_dates,
                                      overall_risk)


@pytest.fixture(scope="session")
def example_instance():
    return load_instance("data/instances/example.dat")


@pytest.fixture(scope="session")
def example_solution():
    return import_solution("data/solutions/example.sol")


def test_makespan(example_solution):
    """Tests the makespan function."""
    assert makespan(example_solution) == 28


def test_weighted_sum_completion_dates(example_instance, example_solution):
    """Tests the weighted sum of completion dates function"""
    assert weighted_sum_completion_dates(
                example_instance,
                example_solution) == 47.699999999999996


def test_overall_risk(example_instance, example_solution):
    """Tests the overall risk function."""
    assert overall_risk(example_instance, example_solution) == 44.5
