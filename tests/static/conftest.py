import pytest
import networkx as nx

from optlis.shared import import_solution
from optlis.static.problem_data import load_instance, Instance


@pytest.fixture(scope="session")
def instance_grid3x3_data():
    """ "The data inside the example.dat instance."""
    nodes = [
        (0, {"type": 0, "p": 0, "q": 1, "r": 0.0}),
        (1, {"type": 1, "p": 5, "q": 0, "r": 0.1}),
        (2, {"type": 1, "p": 5, "q": 0, "r": 0.2}),
        (3, {"type": 1, "p": 5, "q": 0, "r": 0.3}),
        (4, {"type": 1, "p": 5, "q": 0, "r": 0.4}),
        (5, {"type": 1, "p": 5, "q": 0, "r": 0.5}),
        (6, {"type": 1, "p": 5, "q": 0, "r": 0.5}),
        (7, {"type": 1, "p": 5, "q": 0, "r": 0.6}),
        (8, {"type": 1, "p": 5, "q": 0, "r": 0.7}),
    ]
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (3, 6),
        (4, 5),
        (4, 7),
        (5, 8),
        (6, 7),
        (7, 8),
    ]
    return nodes, edges


@pytest.fixture(scope="session")
def instance_grid3x3(instance_grid3x3_data):
    """The example.dat as a Graph instance."""
    G = nx.Graph()
    G.add_nodes_from(instance_grid3x3_data[0])
    G.add_edges_from(instance_grid3x3_data[1])
    return Instance(G)


@pytest.fixture()
def example_instance():
    return load_instance("data/instances/example.dat")


@pytest.fixture()
def example_solution():
    return import_solution("data/solutions/example.sol")
