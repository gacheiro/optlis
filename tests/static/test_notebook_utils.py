import math

from optlis.static.notebook_utils import makespan, overall_risk


def test_makespan(example_solution):
    """Tests the makespan function."""
    assert makespan(example_solution) == 56


def test_overall_risk(example_instance, example_solution):
    """Tests the function to calculate the accumulated risk. Should be equal to
    weighted sum of completion dates.
    """
    assert math.isclose(overall_risk(example_instance, example_solution), 85.2)
