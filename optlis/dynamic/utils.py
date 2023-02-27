import networkx as nx

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
            N=sum(nx.get_node_attributes(self, "kn").values()),
            R=sum(nx.get_node_attributes(self, "kr").values()),
        )

    @property
    def durations(self):
        return list(nx.get_node_attributes(self, "p").values())

    @property
    def risk(self):
        return dict(self._risk)

    @property
    def products(self):
        nproducts = len(self._risk)
        return list(range(nproducts))

    @property
    def initial_concentration(self):
        return dict(self._initial_concentration)

    def degradation_rate(self, p):
        return self._degradation_rate[p]

    def metabolization_rate(self, p, q):
        return self._metabolization_rate[p][q]


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
        nid, ntype, kn, kr, p = line.split()
        nodes.append(
            (
                int(nid),
                {
                    "type": int(ntype),
                    "kn": int(kn),
                    "kr": int(kr),
                    "p": int(p),
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
