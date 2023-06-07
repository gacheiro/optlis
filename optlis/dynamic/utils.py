from typing import Any, Dict, Union, Generator, Tuple, TextIO, Optional
from pathlib import Path

import networkx as nx
import numpy as np

from optlis import Instance as StaticInstance


class Instance(StaticInstance):
    def __init__(
        self, nodes, risk, degradation_rate, metabolization_rate, initial_concentration
    ):
        super().__init__()
        self.add_nodes_from(nodes)
        self._risk = risk
        self._degradation_rate = degradation_rate
        self._metabolization_rate = metabolization_rate
        self._initial_concentration = initial_concentration

    @property
    def resources(self):
        return dict(
            Qn=sum(nx.get_node_attributes(self, "Qn").values()),
            Qc=sum(nx.get_node_attributes(self, "Qc").values()),
        )

    @property
    def durations(self):
        return list(nx.get_node_attributes(self, "p").values())

    @property
    def risk(self):
        try:
            # TODO: deprecate this in favor of the later
            return dict(self._risk)
        except TypeError:
            return np.array(self._risk, dtype=np.float64)

    @property
    def products(self):
        nproducts = len(self._risk)
        return list(range(nproducts))

    def initial_concentration(self, i, p):
        return self._initial_concentration[i][p]

    def degradation_rate(self, p):
        return self._degradation_rate[p]

    def metabolization_rate(self, p, q):
        try:
            # TODO: deprecate this in favor of the later
            return self._metabolization_rate[p][q]
        except KeyError:
            return self._metabolization_rate.get((p, q), 0)

    @property
    def time_periods(self):
        return np.array(range(102))


def load_instance(path):
    """Loads an instance from a file."""
    nodes = []
    risk = {}
    degradation_rate = {}
    metabolization_rate = {}
    initial_concentration = {}

    with open(path, "r") as f:
        lines = f.readlines()

    assert lines[0].startswith("# format: dynamic")
    instance_data = (l for l in lines if not l.startswith("#"))

    nproducts = int(next(instance_data))

    # Parses products' risk
    for _ in range(nproducts):
        line = next(instance_data)
        id_, risk_ = line.split()
        risk[int(id_)] = float(risk_)

    # Parses products' degradation rate
    for _ in range(nproducts):
        line = next(instance_data)
        id_, degradation_rate_ = line.split()
        degradation_rate[int(id_)] = float(degradation_rate_)

    # Parses products'
    for _ in range(nproducts):
        line = next(instance_data)
        id_, *metabolization_rate_ = line.split()
        metabolization_rate[int(id_)] = tuple(float(r) for r in metabolization_rate_)

    nnodes = int(next(instance_data))
    for _ in range(nnodes):
        line = next(instance_data)
        nid, ntype, Qn, Qc, D = line.split()
        nodes.append(
            (
                int(nid),
                {
                    "type": int(ntype),
                    "Qn": int(Qn),
                    "Qc": int(Qc),
                    "D": int(D),
                },
            )
        )

    nconcentration = int(next(instance_data))
    for _ in range(nconcentration):
        line = next(instance_data)
        id_, *initial_concentration_ = line.split()
        initial_concentration[int(id_)] = tuple(
            float(c) for c in initial_concentration_
        )

    instance = Instance(
        nodes, risk, degradation_rate, metabolization_rate, initial_concentration
    )
    instance.time_horizon = int(next(instance_data))

    return instance


def export_instance(instance: Instance, outfile_path: Union[str, Path]) -> None:
    """Exports a problem instance to a file."""
    with open(outfile_path, "w") as outfile:
        _write_instance(instance, outfile)


def _write_instance(instance: Instance, outfile: TextIO) -> None:
    """Writes a problem instance to a file."""
    outfile.write("# format: dynamic\n")

    # Write product risk
    outfile.write(f"{len(instance.products)}\n")
    for pid in instance.products:
        outfile.write(f"{pid} {instance.risk[pid]:.2f}\n")

    # Write product degradation rate
    for pid in instance.products:
        outfile.write(f"{pid} {instance.degradation_rate(pid):.2f}\n")

    # Write product metabolization matrix
    for pid in instance.products:
        outfile.write(f"{pid}")
        for sid in instance.products:
            outfile.write(f" {instance.metabolization_rate(pid, sid):.2f}")
        outfile.write("\n")

    # Write nodes information
    outfile.write(f"{len(instance.nodes)}\n")
    for id, data in instance.nodes(data=True):
        type, Qn, Qc, D = (data["type"], data["Qn"], data["Qc"], data["D"])
        outfile.write(f"{id} {type} {Qn} {Qc} {D}\n")

    # Write initial concentration
    outfile.write(f"{len(instance.nodes)}\n")
    for id in instance.nodes:
        outfile.write(f"{id}")
        for pid in instance.products:
            outfile.write(f" {instance.initial_concentration(id, pid):.2f}")
        outfile.write("\n")

    T = instance.time_periods[-1]
    outfile.write(f"{T}\n")
