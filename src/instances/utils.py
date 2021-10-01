import warnings
from math import ceil
from itertools import groupby

import networkx as nx


class Graph(nx.Graph):
    """Subclass of nx.Graph class with some utility properties."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_horizon = None

    @property
    def origins(self):
        """Returns the list of origins."""
        # Note: is there a better way to do this?
        return [n for n in self.nodes if self.nodes[n]["type"] == 0]

    @property
    def destinations(self):
        """Returns the list of destinations."""
        return [n for n in self.nodes if self.nodes[n]["type"] == 1]

    @property
    def time_periods(self):
        """Returns a list of time periods from 1 to T (inclusive).
           The value of `G.time_horizon` is used for T if it's not None.

           Otherwise, T is calculated with the formula:

           (the graph's diameter * the number of nodes
           + the sum of the job durations) divided by
           the number of wts.
        """
        if self.time_horizon is not None:
            T = self.time_horizon
        else:
            nb_nodes, nb_wts, sum_durations = (
                len(self.nodes),
                sum(nx.get_node_attributes(self, "q").values()),
                sum(nx.get_node_attributes(self, "p").values())
            )
            diameter = nx.diameter(self)
            T = ceil((diameter*nb_nodes + sum_durations) / nb_wts)
        return list(range(1, T+1))

    def dag(self, p=0):
        """Returns a Direct Acyclic Graph representing the
           precedence constraints.
           The `p` param is the relaxation threshold.
        """
        node_risk_dict = nx.get_node_attributes(self, "r")
        for i, ri in node_risk_dict.items():
            for j, rj in node_risk_dict.items():
                if (ri > rj + p
                    and i not in self.origins
                    and j not in self.origins):
                        yield (i, j)


def load_instance(path):
    """Loads an instance from a file."""
    nodes = []
    edges = []
    with open(path, "r") as f:
        nb_nodes = int(f.readline())
        for _ in range(nb_nodes):
            id, type, p, q, r = f.readline().split()
            nodes.append((int(id), {
                "type": int(type),
                "p": int(p),
                "q": int(q),
                "r": float(r),
            }))

        nb_edges = int(f.readline())
        for _ in range(nb_edges):
            i, j = [int(u) for u in f.readline().split()]
            edges.append((i, j))

        G = Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        try:
            T = int(f.readline())
            G.time_horizon = T
        except (EOFError, ValueError):
            warnings.warn("the instance file doesn't provide a time horizon")

    return G


def write_instance(G, outfile):
    """Writes a problem instance to a file"""
    nb_nodes = len(G.nodes)
    nb_edges = len(G.edges)
    outfile.write(f"{nb_nodes}\n")
    for (id, data) in G.nodes(data=True):
        type, p, q, r = (data["type"],
                         data["p"],
                         data["q"],
                         data["r"])
        outfile.write(f"{id} {type} {p} {q} {r:.1f}\n")

    outfile.write(f"{nb_edges}\n")
    for (i, j) in G.edges():
        outfile.write(f"{i} {j}\n")

    T = G.time_periods[-1]
    outfile.write(f"{T}\n")


def export_instance(G, outfile_path):
    """"Exports a problem instance to a text file."""
    with open(outfile_path, "w") as outfile:
        write_instance(G, outfile)


def import_solution(path):
    """Imports and parses a solution file."""
    variables = {}
    with open(path, "r") as sol_file:
        # Discard first line (header)
        _ = sol_file.readline()
        for line in sol_file:
            variable, value = line.split("=")
            try:
                variables[variable.strip()] = int(value)
            except ValueError:
                # special case for the overall_risk variable
                variables[variable.strip()] = float(value)
    return variables


def write_solution(solution, instance_path, outfile):
    """Writes a solution to a text file."""
    outfile.write(f"Solution for instance {instance_path}\n")
    for var, value in solution.items():
        if value and value > 0:
            outfile.write(f"{var} = {value}\n")


def export_solution(solution, instance_path, outfile_path):
    """Exports a solution to a very simple text file because
       pulp's solution files are too big. We only need the
       variables that are > 0 anyway ¯\\_(ツ)_//¯
    """
    with open(outfile_path, "w") as outfile:
        write_solution(solution, instance_path, outfile)
