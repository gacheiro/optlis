from itertools import cycle

import click
import numpy as np
import networkx as nx

from instances.utils import save_instance, Graph


def generate_graph(size=(1, 1), nb_origins=1, q=1, topology="grid"):
    """Generates a graph according to the topology."""
    if topology == "hexagonal":
        g = nx.triangular_lattice_graph(size[0], size[1])
    else:
        g = nx.grid_2d_graph(*size)

    G = nx.convert_node_labels_to_integers(g)

    # Set default attributes for all nodes
    nx.set_node_attributes(G, values=1, name="type")
    nx.set_node_attributes(G, values=1, name="p")
    nx.set_node_attributes(G, values=0, name="q")
    nx.set_node_attributes(G, values=0.5, name="r")

    # Updates the values for the origins
    for i in range(nb_origins):
        G.nodes[i]["type"] = 0
        G.nodes[i]["q"] = q
        G.nodes[i]["p"] = 0
        G.nodes[i]["r"] = 0

    return G


def generate_instance(size=(1, 1), nb_origins=1, q=1, topology="grid",
                      risk_distribution="homogeneous", seed=0):
    """Generates a problem instance."""
    G = generate_graph(size, nb_origins, q, topology)
    n = len(G.nodes)

    if risk_distribution == "bipartite":
        # Updates the risk with values {0.5, 0.6} interpolated
        for i, r in zip(range(1, n), 
                        cycle([0.5, 0.6])):
            G.nodes[i]["r"] = r
            
    elif risk_distribution == "uniform":
        np.random.seed(seed)
        # Update the risk with values over an uniform distribution (0, 1)
        for i, r in zip(range(1, n), 
                        np.random.uniform(0.1, 1, n)):
            G.nodes[i]["r"] = r
    
    return Graph(G)


def grid(size=(3, 3), nb_origins=1, p=1, q=1, r=0.5):
    """Generates a random instance in a 2d grid format."""
    gridG = nx.grid_2d_graph(*size)
    G = nx.convert_node_labels_to_integers(gridG)
    # Set default attributes
    nx.set_node_attributes(G, values=0, name="p")
    nx.set_node_attributes(G, values=0, name="q")
    nx.set_node_attributes(G, values=0.0, name="r")
    # Set random attributes for every kind of node
    nb_nodes = size[0]*size[1]
    for i in range(nb_origins):
        G.nodes[i]["type"] = 0
        G.nodes[i]["q"] = q
    for i in range(nb_origins, nb_nodes):
        G.nodes[i]["type"] = 1
        G.nodes[i]["p"] = p
        G.nodes[i]["r"] = r
    return Graph(G)


def grid_uniform(size, nb_origins=1, p=1, q=1, seed=None):
    """Generates a grid instance where the r_i attribute is randomly
       chosen from the uniform distribution [0.1, 1).
    """
    graph = grid(size, nb_origins, p, q)
    # Updates the r attributes with the random uniform distribution
    nb_nodes = size[0]*size[1] - nb_origins
    nb_destinations = nb_nodes - nb_origins

    np.random.seed(seed)
    r_values = np.random.uniform(0.1, 1, nb_destinations)
    for i, r in zip(range(nb_origins, nb_nodes), r_values):
        graph.nodes[i]["r"] = r
    return graph


def hexagonal(size=(3, 3), nb_origins=1, p=1, q=1, r=0.5):
    """Generates an instance over a hexagonal grid layout."""
    hG = nx.triangular_lattice_graph(size[0], size[1])
    G = nx.convert_node_labels_to_integers(hG)
    # Set default attributes
    nx.set_node_attributes(G, values=0, name="p")
    nx.set_node_attributes(G, values=0, name="q")
    nx.set_node_attributes(G, values=0.0, name="r")

    nb_nodes = len(G.nodes)
    for i in range(nb_origins):
        G.nodes[i]["type"] = 0
        G.nodes[i]["q"] = q
    for i in range(nb_origins, nb_nodes):
        G.nodes[i]["type"] = 1
        G.nodes[i]["p"] = p
        G.nodes[i]["r"] = r
    return Graph(G)


@click.command()
@click.option("--size", type=(int, int), default=(3, 3),
              help="The size of the grid (m, n).")
@click.option("--norigins", default=1, help="The number of origins.")
@click.option("-p", type=int, default=1,
              help="The value of p attributes for each destination ex. -p 1.")
@click.option("-q", type=int, default=1,
              help="The range of q attributes for each origin ex. -q 1.")
@click.option("-r", help="The value of r attributes for each destination ex. -r 0.5 "
                   "or -r uniform (default)", default="uniform")
@click.option("--path", help="The path to save the instance. If not provided, print the"
                        "output to the stdout.")
def generate(size, norigins, p, q, r, path):
    if r == "uniform":
        graph = grid_uniform(size, norigins, p, q)
    else:
        graph = grid(size, norigins, p, q, float(r))
    save_instance(graph, path)


if __name__ == "__main__":
    generate()
