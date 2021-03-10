from pathlib import Path

import pytest
import networkx as nx

from instances.inst import loads, grid2d


@pytest.fixture(scope="session")
def instance_grid3x3_data():
    nodes = [
        (0, {"type": 0, "p": 0, "q": 1, "r": 0.0}),
        (1, {"type": 0, "p": 0, "q": 1, "r": 0.0}),
        (2, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
        (3, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
        (4, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
        (5, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
        (6, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
        (7, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
        (8, {"type": 1, "p": 1, "q": 0, "r": 0.5}),
    ]
    edges = [
        (0, 1), (0, 3),
        (1, 2), (1, 4),
        (2, 5),
        (3, 4), (3, 6),
        (4, 5), (4, 7),
        (5, 8),
        (6, 7),
        (7, 8),
    ]
    return nodes, edges


@pytest.fixture(scope="session")
def instance_grid3x3(instance_grid3x3_data):
    G = nx.Graph()
    G.add_nodes_from(instance_grid3x3_data[0])
    G.add_edges_from(instance_grid3x3_data[1])
    return G


def test_loads(instance_grid3x3_data):
    """Tests the function to load an instance from a file."""
    G = loads(Path("data/instances/example.dat"))
    assert list(G.nodes(data=True)) == instance_grid3x3_data[0]
    assert set(G.edges) == set(instance_grid3x3_data[1])


def test_grid2d(instance_grid3x3_data):
    """Tests the function to generate grid instances.
       It tries to generate the same `grid3x3.dat` instance.
    """
    G = grid2d()
    assert list(G.nodes(data=True)) == instance_grid3x3_data[0]
    assert set(G.edges) == set(instance_grid3x3_data[1])
