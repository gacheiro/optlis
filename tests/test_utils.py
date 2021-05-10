from pathlib import Path

from instances import load_instance


def test_loads(instance_grid3x3_data):
    """Tests the function to load an instance from a file."""
    G = load_instance(Path("data/instances/example.dat"))
    assert list(G.nodes(data=True)) == instance_grid3x3_data[0]
    assert set(G.edges) == set(instance_grid3x3_data[1])


def test_Graph():
    """Tests the Graph class."""
    G = load_instance(Path("data/instances/example.dat"))
    assert list(G.origins) == [0, 1]
    assert list(G.destinations) == [2, 3, 4, 5, 6, 7, 8]
    assert list(G.time_periods)[-1] == 36 # NOTE: see G.time_periods
    assert set(G.precedencies) == {       # docstring for the formula
        (8, 7), (7, 6), (7, 5),
        (6, 4), (5, 4), (4, 3), (3, 2),
    }
