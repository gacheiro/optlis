from typing import Any, Dict, Union, Generator, Tuple, TextIO, Optional

import warnings
from math import ceil
from pathlib import Path
from functools import cached_property

import networkx as nx  # type: ignore
import numpy as np
import numpy.typing as npt

from optlis.shared import set_product
from optlis.static.models.ctypes import c_instance, c_int32, c_size_t, c_double, POINTER


class Instance(nx.Graph):
    """Subclass of nx.Graph class with some utility properties."""

    time_horizon: Optional[int] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # NOTE: some of these methods should return either a read-only np.array or a new copy
    @cached_property
    def depots(self) -> npt.NDArray[np.int32]:
        """Returns the list of depots."""
        return np.array(
            [n for n in self.nodes if self.nodes[n]["type"] == 0], dtype=np.int32
        )

    @cached_property
    def tasks(self) -> npt.NDArray[np.int32]:
        """Returns the list of tasks."""
        return np.array(
            [n for n in self.nodes if self.nodes[n]["type"] == 1], dtype=np.int32
        )

    @property
    def time_periods(self) -> npt.NDArray[np.int32]:
        """Returns a list of time units from 0 to T - 1.
        The `G.time_horizon` attribute is used for T if it's not None.
        Otherwise, T is calculated by the formula:

        (the graph's diameter * the number of nodes
        + the sum of the job durations) divided by
        the number of wts + 1.
        """
        if self.time_horizon is not None:
            T = self.time_horizon
        else:
            nb_nodes, nb_wts, sum_durations = (
                len(self.nodes),
                sum(nx.get_node_attributes(self, "q").values()),
                sum(nx.get_node_attributes(self, "p").values()),
            )
            diameter = nx.diameter(self)
            T = ceil((diameter * nb_nodes + sum_durations) / nb_wts) + 1
            T = max(T, 21)  # sets a minimum of 20 time units
        return np.array(list(range(T)), dtype=np.int32)

    @cached_property
    def setup_times(self) -> npt.NDArray[Any]:
        """Returns a 2d numpy array with the sequence-dependent setup times."""
        s_dict = dict(nx.shortest_path_length(self, weight="weight"))
        nnodes = len(self.nodes())
        s = np.zeros((nnodes, nnodes), dtype=np.int32)
        for i in self.nodes():
            for j in self.nodes():
                s[i][j] = s_dict[i][j]
        return s

    @cached_property
    def node_resources(self) -> npt.NDArray[np.int32]:
        """Returns a numpy array with the number of resources present at each node."""
        return np.array(
            list(nx.get_node_attributes(self, "q").values()), dtype=np.int32
        )

    @cached_property
    def node_durations(self) -> npt.NDArray[np.int32]:
        """Returns a numpy array with the `duration` attribute of each node.
        NOTE: Some nodes may have 0 duration, which means they are not tasks.
        """
        return np.array(
            list(nx.get_node_attributes(self, "p").values()), dtype=np.int32
        )

    @cached_property
    def node_risks(self) -> npt.NDArray[np.float64]:
        """Returns a numpy array with the `risk` attribute of each node.

        NOTE: Some nodes may have 0 risk, which means they are not tasks.
        """
        return np.array(
            list(nx.get_node_attributes(self, "r").values()), dtype=np.float64
        )

    def precedence(self, d: float = 0) -> Generator[Tuple[int, int], None, None]:
        """Generates the set of priority relations for a given relaxation threshold
        in the form of (i, j) tuples.

        NOTE: use d = 0, 0.5 and 1 for the strict, moderate and none priority rules,
              respectively.
        """
        node_risk_dict = nx.get_node_attributes(self, "r")
        nnodes = len(self.nodes())
        adj_m = np.zeros((nnodes, nnodes), dtype=np.int32)

        # Creates a DAG with all the priority constraints
        for i, ri in node_risk_dict.items():
            for j, rj in node_risk_dict.items():
                if ri > rj + d and i not in self.depots and j not in self.depots:
                    adj_m[i][j] = 1

        # Removes redundant arcs: if the priorities i -> k, k -> j and i -> j
        # exist, remove i -> j because it's redundant
        for i, k, j in set_product(self.nodes, self.nodes, self.nodes):
            if adj_m[i][k] == 1 and adj_m[k][j] == 1 and adj_m[i][j] == 1:
                adj_m[i][j] = 0

        # Returns the reduced DAG
        for i, j in set_product(self.nodes, self.nodes):
            if adj_m[i][j] == 1:
                yield (i, j)


    def c_struct(self) -> c_instance:
        return c_instance(
            c_size_t(len(self.nodes)),
            c_size_t(len(self.tasks)),
            c_size_t(sum(self.node_resources)),
            self.tasks.ctypes.data_as(POINTER(c_int32)),
            self.node_durations.ctypes.data_as(POINTER(c_int32)),
            self.node_risks.ctypes.data_as(POINTER(c_double)),
            self.setup_times.ctypes.data_as(POINTER(c_int32)),
        )


def load_instance(path: Union[str, Path], use_setup_times: bool = True) -> Instance:
    """Loads an instance from a file."""
    nodes = []
    edges = []
    with open(path, "r") as f:
        nb_nodes = int(f.readline())
        for _ in range(nb_nodes):
            id, type, p, q, r = f.readline().split()
            nodes.append(
                (
                    int(id),
                    {
                        "type": int(type),
                        "p": int(p),
                        "q": int(q),
                        "r": float(r),
                    },
                )
            )

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


def export_instance(instance: Instance, outfile_path: Union[str, Path]) -> None:
    """Exports a problem instance to a file."""
    with open(outfile_path, "w") as outfile:
        _write_instance(instance, outfile)


def _write_instance(instance: Instance, outfile: TextIO) -> None:
    """Writes a problem instance to a file."""
    nb_nodes = len(instance.nodes)
    nb_edges = len(instance.edges)
    outfile.write(f"{nb_nodes}\n")

    for (id, data) in instance.nodes(data=True):
        type, p, q, r = (data["type"], data["p"], data["q"], data["r"])
        outfile.write(f"{id} {type} {p} {q} {r:.1f}\n")

    outfile.write(f"{nb_edges}\n")
    for (i, j) in instance.edges():
        outfile.write(f"{i} {j}\n")

    T = instance.time_periods[-1]
    outfile.write(f"{T}\n")
