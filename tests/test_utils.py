from pathlib import Path

from instances import loads


def test_loads(instance_grid3x3_data):
    """Tests the function to load an instance from a file."""
    G = loads(Path("data/instances/example.dat"))
    assert list(G.nodes(data=True)) == instance_grid3x3_data[0]
    assert set(G.edges) == set(instance_grid3x3_data[1])


def test_Graph():
    """Test the Graph class."""
    G = loads(Path("data/instances/example.dat"))
    assert list(G.origins) == [0, 1]
    assert list(G.destinations) == [2, 3, 4, 5, 6, 7, 8]
    assert list(G.time_periods) == list(range(1, 26)) # NOTE: see G.time_periods
    assert set(G.precedencies) == {                   # docstring for the formula
        (8, 7), (7, 6), (7, 5),
        (6, 4), (5, 4), (4, 3), (3, 2),
    }
