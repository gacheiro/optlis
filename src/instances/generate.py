import random
from math import log, floor

from scipy.stats import triang
import click
import networkx as nx

from instances.utils import save_instance, Graph


def grid2d(size=(3, 3), nb_origins=1, p=1, q=1, r=0.5):
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


def grid_triangular_rdistribution(size, c, nb_origins=1, p=1, q=1):
    """Generates a grid instance where the r_i attribute is randomly
       generated over a triangular distribution, where c is the mode.

       Suggested values for c: 0.1 for positive, 0.5 for zero or 0.9 for
       negative skewness.
    """
    graph = grid2d(size, nb_origins, p, q)
    # Updates the r attributes with the random triangular values
    nb_nodes = size[0]*size[1]
    r_values = triang.rvs(c, size=(nb_nodes - nb_origins))
    for i, r in zip(range(nb_origins, nb_nodes), r_values):
        graph.nodes[i]["r"] = r
    return graph


@click.command()
@click.option("--size", type=(int, int), default=(3, 3),
              help="The size of the grid (m, n).")
@click.option("--norigins", default=1, help="The number of origins.")
@click.option("-p", type=int, default=1,
              help="The value of p attributes for each destination ex. -p 1.")
@click.option("-q", type=int, default=1,
              help="The range of q attributes for each origin ex. -q 1.")
@click.option("-r", help="The value of r attributes for each destination ex. -r 0.5 "
                   "or the skewness ex. -r positive, -r zero, -r negative")
@click.argument("path")
def generate(size, norigins, p, q, r, path):
    if r == "positive":
        graph = grid_triangular_rdistribution(size, 0.1, norigins, p, q)
    elif r == "zero":
        graph = grid_triangular_rdistribution(size, 0.5, norigins, p, q)
    elif r == "negative":
        graph = grid_triangular_rdistribution(size, 0.9, norigins, p, q)
    else:
        graph = grid2d(size, norigins, p, q, float(r))
    save_instance(graph, path)


if __name__ == "__main__":
    generate()
