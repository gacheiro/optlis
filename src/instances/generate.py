import random

import click
import networkx as nx

from instances.utils import save, Graph


def grid2d(size=(3, 3),
           nb_origins=2,
           p_range=(1, 1),
           q_range=(1, 1),
           r_range=(0.5, 0.5),
           randf=random.uniform,
           randi=random.randint):
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
        G.nodes[i]["q"] = randi(*q_range)
    for i in range(nb_origins, nb_nodes):
        G.nodes[i]["type"] = 1
        G.nodes[i]["p"] = randi(*p_range)
        G.nodes[i]["r"] = randf(*r_range)
    return Graph(G)


@click.command()
@click.option("--size", type=(int, int), default=(3, 3),
              help="The size of the grid (m, n).")
@click.option("--norigins", default=2, help="The number of origins.")
@click.option("-p", type=(int, int), default=(1, 1),
              help="The range of the p attribute ex. -p 1 5.")
@click.option("-q", type=(int, int), default=(1, 1),
              help="The range of the q attribute ex. -q 1 2.")
@click.option("-r", type=(float, float), default=(0.5, 0.5),
              help="The range of the q attribute ex. -r 0.1 1.0.")
@click.argument("path")
def generate(size, norigins, p, q, r, path):
    G = grid2d(size, norigins, p, q, r)
    save(G, path)


if __name__ == "__main__":
    generate()
