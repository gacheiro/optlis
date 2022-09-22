import warnings
from math import ceil
from functools import cached_property

import networkx as nx
import numpy as np


class Instance(nx.Graph):
    """Subclass of nx.Graph class with some utility properties."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_horizon = None

    @cached_property
    def depots(self):
        """Returns the list of origins."""
        # Note: is there a better way to do this?
        return [n for n in self.nodes if self.nodes[n]["type"] == 0]

    @cached_property
    def tasks(self):
        """Returns the list of destinations."""
        return [n for n in self.nodes if self.nodes[n]["type"] == 1]

    @property
    def time_periods(self):
        """Returns a list of time periods from 0 to T - 1.
           The `G.time_horizon` attribute is used for T if it's not None.
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
        return list(range(T))

    @cached_property
    def setup_times(self):
        return dict(nx.shortest_path_length(self, weight="weight"))

    @cached_property
    def task_durations(self):
        return nx.get_node_attributes(self, "p")

    @cached_property
    def task_risks(self):
        # NOTE: make this immutable
        return np.array([r for r in nx.get_node_attributes(self, "r").values()])

    def precedence(self, d=0):
        """Generates the set precedence for a given relaxation threshold in the form of
           (i, j) tuples.

           Note: use d = 0, 0.5 and 1 for the strict, moderate and none priority rules,
                 respectively.
        """
        node_risk_dict = nx.get_node_attributes(self, "r")
        for i, ri in node_risk_dict.items():
            for j, rj in node_risk_dict.items():
                if (ri > rj + d
                    and i not in self.depots
                    and j not in self.depots):
                        yield (i, j)


def load_instance(path, use_setup_times=True):
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

        instance = Instance()
        instance.add_nodes_from(nodes)

        # Enable or disable sequence-dependent setup times
        weight = 1 if use_setup_times else 0
        instance.add_edges_from(edges, weight=weight)

        try:
            T = int(f.readline())
            instance.time_horizon = T
        except (EOFError, ValueError):
            warnings.warn("the instance file doesn't provide a time horizon")
    return instance


def export_instance(G, outfile_path):
    """"Exports a problem instance to a file."""
    with open(outfile_path, "w") as outfile:
        _write_instance(G, outfile)


def _write_instance(G, outfile):
    """Writes a problem instance to a file."""
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


def import_solution(path):
    """Imports a solution from a file."""
    variables = {}
    with open(path, "r") as sol_file:
        # Discard first line (header)
        _ = sol_file.readline()
        for line in sol_file:
            variable, value = line.split("=")
            try:
                variables[variable.strip()] = int(value)
            except ValueError:
                # Special case for the overall_risk variable
                variables[variable.strip()] = float(value)
    return variables


def export_solution(solution, instance_path, outfile_path):
    """Exports a solution to a simple text file since
       pulp's solution files are too big. We only need the
       variables that are > 0 anyway ¯\\_(ツ)_//¯

       Notes: only works for pulp's solution and
              variables with value 0 are ignored.
    """
    with open(outfile_path, "w") as outfile:
        _write_solution(solution, instance_path, outfile)


def _write_solution(solution, instance_path, outfile):
    """Writes a solution to a file."""
    outfile.write(f"Solution for instance {instance_path}\n")
    for var, value in solution.items():
        if value and value > 0:
            outfile.write(f"{var} = {value}\n")
