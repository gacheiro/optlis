from itertools import cycle
from math import ceil

import pulp as plp
import networkx as nx
import click
import matplotlib.pyplot as plt

from instances import loads


def make_prob(G):
    """Implements a model based on Christofides et al (1987) for the RCPSP."""
    V = G.nodes()
    # The set of origins
    O = list(n for n in G.nodes() if G.nodes[n]["type"] == 0)
    # The set of 'jobs' to process
    D = list(n for n in G.nodes() if G.nodes[n]["type"] == 1)
    # The number of wt at each origin
    q = nx.get_node_attributes(G, "q")
    # The duration of each job
    p = nx.get_node_attributes(G, "p")
    # The risk at each destination
    r = nx.get_node_attributes(G, "r")
    # The distance between every pair of nodes
    d = dict(nx.shortest_path_length(G))
    # The estimated amount of time periods to process all jobs (an upper bound)
    # indexed from 1 to T
    T = G.time_periods

    prob = plp.LpProblem("Cmax_Pulse_Flow", plp.LpMinimize)

    # Creates the model's variables
    Cmax = plp.LpVariable("Cmax", lowBound=0, cat=plp.LpInteger)
    S = plp.LpVariable.dicts("S", indexs=D, lowBound=0, cat=plp.LpInteger)
    C = plp.LpVariable.dicts("C", indexs=D, lowBound=0, cat=plp.LpInteger)
    x = plp.LpVariable.dicts("x", indexs=(V, V, T), cat=plp.LpBinary)

    prob.vars = {
        "Cmax": Cmax,
        "S": S,
        "C": C,
        "x": x,
    }

    # Add the objective function to 'prob'
    prob += Cmax, "Makespan"

    # Flow depart from origins
    for i in O:
        prob += (plp.lpSum([x[i][j][t] for t in T
                                       for j in D]) <= q[i],
                 f"Flow_depart_from_origin_{i}")

    '''
    # Ensure the maximum number of resources is respected
    for t in T:
        prob += (plp.lpSum([x[i][j][tau] for j in D
                                         for i in V
                                         for tau in range(t - p[i] + 1, t+1)
                                         if tau >= 1])
                <= sum(q.values()),
                f"Resource_constraint_at_period_{t}")
    '''

    # Flow conservation constraints
    for j in D:
        for t in T:
            prob += (plp.lpSum([x[i][j][t] for i in V])
                     - plp.lpSum([x[j][i][t + p[j] + d[j][i]] for i in V if t + p[j] + d[j][i] <= T[-1]]) == 0,
                     f"Flow_conservation_on_{j}_at_periods_{t}")

    # Every destination has to be cleaned at some point (1)
    for j in D:
        prob += (plp.lpSum([x[i][j][t] for t in T
                                       for i in V]) == 1,
                 f"Clean_destination_arrive_{j}")

    # Every destination has to be cleaned at some point (2)
    for j in D:
        prob += (plp.lpSum([x[j][i][t] for t in T
                                       for i in V]) == 1,
                 f"Clean_destination_depart_{j}")

    # Calculates the start time of every node
    for j in D:
        prob += (S[j] - plp.lpSum([t*x[i][j][t] for t in T
                                                for i in V]) == 0,
                 f"Start_time_of_{j}")

    # Calculates the completion time of every node
    for j in D:
        prob += C[j] - S[j] == p[j], f"Completion_time_of_{j}"

    # Calculates the makespan
    for j in D:
        prob += Cmax - C[j] >= 0, f"Cmax_geq_C_{j}"

    # Add precedence constraints
    for i, j in G.precedencies:
            prob += S[i] <= S[j], f"Clean {i} before {j}"

    # The problem is written to an .lp file
    prob.writeLP("CmaxPulseFlow.lp")
    return prob


def plot_prob(G, prob):
    """Plots the solution."""
    Cmax, C, x = (prob.vars["Cmax"],
                  prob.vars["C"],
                  prob.vars["x"])

    # The risk at each destination
    r = nx.get_node_attributes(G, "r")

    # Config graph style
    options = {
        "edgecolors": "tab:gray",
        "alpha": 1,
        "node_size": [100 + 300*ri for ri in r.values()],
        "font_color": "white",
        "font_size": 10,
        "with_labels": True,
    }
    colors = []
    for node, data in G.nodes(data=True):
        if data["type"] == 0:
            colors.append("tab:blue")
        else:
            colors.append("tab:red")

    # Create a NxM subplots
    figsize = (14, 15)
    rows, cols = 5, 4
    axes = plt.figure(figsize=figsize,
                      constrained_layout=True).subplots(rows, cols)
    axs = axes.flat

    # Draws the whole graph at first ax
    pos = nx.spring_layout(G)
    nx.draw(G, pos, axs[0], node_color=colors, **options)

    # Select periods with active flow between nodes
    active_periods = sorted({t for t in G.time_periods
                               for i in G.nodes
                               for j in G.nodes
                               if x[i][j][t].varValue == 1})

    # The paths between every pair of nodes
    # Used to display the flows
    paths = dict(nx.all_pairs_shortest_path(G))

    # We use a DiGraph to display the solution
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes(data=True))

    for ax, active_t in zip(axs[1:], active_periods):
        ax.set_title(f"t={active_t}")

        # Discover which destinations were already cleaned
        for node in G.destinations:
            if C[node].varValue <= active_t:
                colors[node] = "tab:green"

        # Active flow between nodes
        sources_sinks = [(i, j) for i in DG.nodes
                                for j in DG.nodes
                                if x[i][j][active_t].varValue == 1]

        # Draws each active flow with a diff style
        styles = cycle(["solid", "dashed", "dashdot"])
        for i, j in sources_sinks:
            path = paths[i][j]
            nx.draw_networkx_edges(
                DG,
                pos,
                ax=ax,
                style=next(styles),
                edgelist=list(zip(path, path[1:])),
            )

        # Finally, draws the nodes
        nx.draw_networkx(
            DG,
            pos,
            ax=ax,
            node_color=colors,
            **options,
        )


@click.command()
@click.argument("path")
def run(path):
    """Runs the model from command line."""
    G = loads(path)
    prob = make_prob(G)
    # Solve the problem with CPLEX
    # Comment this following line to use Pulp's builtin solver
    solver = plp.getSolver('CPLEX_CMD')
    prob.solve(solver)

    # Print variables with it's resolved optimum value
    for v in prob.variables():
        if v.varValue > 0:
            print(v.name, "=", v.varValue)

    # Print the optimum objective function value   
    print("Makespan = ", plp.value(prob.objective))
#    plot_prob(G, prob)
#    plt.show()


if __name__ == "__main__":
    run()
