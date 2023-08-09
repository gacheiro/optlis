from typing import Union, Tuple, Dict, Any

from pathlib import Path

import numpy as np
import networkx as nx  # type: ignore

from optlis.dynamic.problem_data import Instance, export_instance


def _graph(size: Tuple[int, int] = (1, 1), res: Tuple[int, int] = (1, 1)) -> Instance:
    """Generates a problem instance."""
    lattice = nx.triangular_lattice_graph(size[0], size[1])
    g = nx.convert_node_labels_to_integers(lattice)

    # Set default attributes for all nodes
    nx.set_node_attributes(g, values=1, name="type")
    nx.set_node_attributes(g, values=0, name="D")
    nx.set_node_attributes(g, values=0, name="Qn")
    nx.set_node_attributes(g, values=0, name="Qc")

    # Updates the depot's attributes
    g.nodes[0]["type"] = 0
    g.nodes[0]["Qn"] = res[0]
    g.nodes[0]["Qc"] = res[1]

    return g


def two_species_instance(size, res, zero_degradation_rate=False, random_seed=0):
    """Generates a random two species model instance."""
    g = _graph(size, res)

    nnodes = len(g.nodes)
    ntasks = len([n for n in g.nodes if g.nodes[n]["type"] == 1])
    rng = np.random.default_rng(random_seed)

    nproducts = 3
    products = (0, 1, 2)
    risk = rng.uniform(0.1, 1, nproducts)
    risk[0] = 0

    # Generates `|V|` random initial concentration for each product specie
    initial_concentration = rng.normal(0.5, 0.5, (nnodes, nproducts))
    for i in range(nnodes):
        for j in range(nproducts):
            if j == 0:
                initial_concentration[i][j] = 0

            elif initial_concentration[i][j] < 0:
                initial_concentration[i][j] = 0

    if zero_degradation_rate:
        degradation_rate = np.zeros(nproducts)
    else:
        degradation_rate = rng.uniform(0.01, 0.05, nproducts)

    metabolization_map = {(products[1], products[2]): rng.uniform(0.01, 0.05)}

    return Instance(
        g.nodes(data=True),
        risk,
        degradation_rate,
        metabolization_map,
        initial_concentration,
    )


def decrease(graph_size, nresources=(0, 0), random_seed=0):
    g = _graph(graph_size, nresources)

    nnodes = len(g.nodes)
    ntasks = len([n for n in g.nodes if g.nodes[n]["type"] == 1])
    rng = np.random.default_rng(random_seed)

    nproducts = 3
    products = (0, 1, 2)  # product 0 is neutral
    risk = [0, 1, 0.5]

   # Generates `|V|` random initial concentration for each product specie
    parent_initial_concentration = rng.normal(0.5, 0.5, nnodes)
    metabolite_initial_concentration = rng.normal(0.5, 0.5, nnodes)
    initial_concentration = np.zeros(shape=(nnodes, nproducts))
    for i in range(nnodes):
        initial_concentration[i][1] = max(0, parent_initial_concentration[i])
        initial_concentration[i][2] = max(0, metabolite_initial_concentration[i])

    # degradation_rate = rng.uniform(0.01, 0.03)
    # metabolization_rate = rng.uniform(0.03, 0.08)
    degradation_rate = 0.008
    metabolization_rate = 0.01

    metabolization_map = {
        (products[1], products[2]): metabolization_rate
    }  # 1 --> 2 at a rate

    yield f"hx-n{ntasks}-ab-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, 0, 0],
        metabolization_map,
        initial_concentration,
    )

    yield f"hx-n{ntasks}-abs-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, 0, degradation_rate],  # 2 --> sink at a rate
        metabolization_map,
        initial_concentration,
    )

    yield f"hx-n{ntasks}-asb-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, degradation_rate, 0],  # 1 --> sink at a rate
        metabolization_map,
        initial_concentration,
    )

    yield f"hx-n{ntasks}-asbs-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, degradation_rate, degradation_rate],  # 1, 2 --> sink at a rate
        metabolization_map,
        initial_concentration,
    )



def increase(graph_size, nresources=(0, 0), random_seed=0):
    g = _graph(graph_size, nresources)

    nnodes = len(g.nodes)
    ntasks = len([n for n in g.nodes if g.nodes[n]["type"] == 1])
    rng = np.random.default_rng(random_seed)

    nproducts = 3
    products = (0, 1, 2)  # product 0 is neutral
    risk = [0, 1, 0.5]

    # Generates `|V|` random initial concentration for each product specie
    parent_initial_concentration = rng.normal(0.5, 0.5, nnodes)
    metabolite_initial_concentration = rng.normal(0.5, 0.5, nnodes)
    initial_concentration = np.zeros(shape=(nnodes, nproducts))
    for i in range(nnodes):
        initial_concentration[i][2] = max(0, parent_initial_concentration[i])
        initial_concentration[i][1] = max(0, metabolite_initial_concentration[i])

    # degradation_rate = rng.uniform(0.01, 0.03)
    # metabolization_rate = rng.uniform(0.03, 0.08)
    degradation_rate = 0.008
    metabolization_rate = 0.01

    metabolization_map = {
        (products[2], products[1]): metabolization_rate
    }  # 2 --> 1 at a rate

    yield f"hx-n{ntasks}-ba-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, 0, 0],
        metabolization_map,
        initial_concentration,
    )

    yield f"hx-n{ntasks}-bsa-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, 0, degradation_rate],  # 2 --> sink at a rate
        metabolization_map,
        initial_concentration,
    )

    yield f"hx-n{ntasks}-bas-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, degradation_rate, 0],  # 1 --> sink at a rate
        metabolization_map,
        initial_concentration,
    )

    yield f"hx-n{ntasks}-bsas-q{nresources[0]}-{nresources[1]}", Instance(
        g.nodes(data=True),
        risk,
        [0, degradation_rate, degradation_rate],  # 1, 2 --> sink at a rate
        metabolization_map,
        initial_concentration,
    )


def generate_benchmark(export_dir: Union[str, Path] = "", random_seed: int = 0) -> None:
    """Generate the instance benchmark."""
    # Generates graphs with n = 9, 17, 33 (1 depot + n-1 tasks)
    for bname in ("increase", "decrease"):

        _export_dir = export_dir / bname
        _export_dir.mkdir(parents=True, exist_ok=True)

        if bname == "increase":
            generate = increase
        elif bname == "decrease":
            generate = decrease

        for graph_size in [(5, 1), (2, 9), (4, 11), (9, 11)]:
            for name, instance in generate(
                graph_size, nresources=(0, 0), random_seed=random_seed
            ):
                export_instance(
                    instance, Path(_export_dir / f"{name}.dat")
                )

            # Gets the amount of nodes generated by the `size` of the hex grid
            # which should be n = 9, 17, 33 (1 depot + n-1 tasks)
            n = len(instance.nodes)

            # Generates instances with 2^0, 2^1, ..., 2^log_2(n-1) teams
            for q in [2**i for i in range(10) if 2**i < n - 1]:

                for name, instance in generate(
                    graph_size, nresources=(q, q), random_seed=random_seed
                ):
                    export_instance(
                        instance, Path(_export_dir / f"{name}.dat")
                    )


def from_command_line(args: Dict[str, Any]) -> None:
    generate_benchmark(args["export-dir"], args["seed"])
