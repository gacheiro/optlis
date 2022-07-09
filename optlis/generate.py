import argparse
from pathlib import Path
from itertools import cycle

import numpy as np
import networkx as nx

from optlis.utils import Graph, export_instance


def generate_graph(size=(1, 1), nb_origins=1, q=1, topology="grid"):
    """Generates a graph according to the topology."""
    if topology == "hexagonal":
        g = nx.triangular_lattice_graph(size[0], size[1])
    else:
        # Grid is the default topology
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
                      risk_distribution="homogeneous",
                      job_duration_distribution="homogeneous", seed=0):
    """Generates a problem instance with different configurations of
       graph topology, risk and job duration distribution.
    """
    G = generate_graph(size, nb_origins, q, topology)
    n = len(G.nodes)

    if risk_distribution == "bipartite":
        # Updates the risk with values {0.5, 0.6} interpolated
        for i, r in zip(range(1, n), 
                        cycle([0.5, 0.6])):
            G.nodes[i]["r"] = r
            
    elif risk_distribution == "uniform":
        rng = np.random.default_rng(seed)
        # Update the risk with values over an uniform distribution [0.1, 1]
        for i, r in zip(range(1, n), 
                        rng.uniform(0.1, 1, n)):
            G.nodes[i]["r"] = r
    
    if job_duration_distribution == "uniform":
        rng = np.random.default_rng(seed)
        # Update the job duration with values over an uniform distribution [1, 11)
        for i, p in zip(range(1, n), 
                        rng.integers(1, 11, n)):
            G.nodes[i]["p"] = p

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


def hex_grid_instances(export_dir="", seed=0):
    """Generate the set of instances over a hexagonal grid layout
       with homogeneous and uniformly distributed risk and job duration.
    """
    # for rdist in ["uniform"]:
        # for pdist in ["homogeneous", "uniform"]:
    rdist = "uniform"
    pdist = "uniform"
    # Generates graphs with 9, 17, 33, 65 nodes (n-1 nodes)
    for size in [(5, 1), (2, 9), (4, 11), (9, 11)]:
        G = generate_instance(
            size,
            nb_origins=1,
            q=1,
            topology="hexagonal",
            risk_distribution=rdist,
            job_duration_distribution=pdist,
            seed=seed
        )

        # We get the amount of nodes generated by the `size` of the hex grid
        # which should be 9, 17, 33, 65, 129 nodes
        n = len(G.nodes)
        export_instance(
            G,
            Path(f"{export_dir}/hx-n{n-1}-p{pdist[0]}-r{rdist[0]}-q{1}.dat")
        )

        # Generates instances with work-troops from 1 to n-1 in powers of 2
        for q in [2**i for i in range(1, 10) if 2**i < n] + [n-1]:
            G = generate_instance(
                size,
                nb_origins=1,
                q=q,
                topology="hexagonal",
                risk_distribution=rdist,
                job_duration_distribution=pdist,
                seed=0
            )

            export_instance(
                G,
                Path(f"{export_dir}/hx-n{n-1}-p{pdist[0]}-r{rdist[0]}-q{q}.dat")
            )


def from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("export-dir", type=Path,
                        help="directory to export instances (must exist)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random number generator (default 0)")
    args = vars(parser.parse_args())
    hex_grid_instances(args["export-dir"],
                       args["seed"])