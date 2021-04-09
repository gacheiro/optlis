from itertools import cycle

import pulp as plp
import networkx as nx
import click
import matplotlib.pyplot as plt

from instances import loads


def make_prob(G):
    """Implements the linear model for the problem."""
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
    c = dict(nx.shortest_path_length(G))
    # The estimated amount of time periods to process all jobs (an upper bound)
    # indexed from 1 to T
    T = G.time_periods

    # Creates the model's variables
    makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpInteger)
    sd = plp.LpVariable.dicts("sd", indexs=D, lowBound=0, cat=plp.LpInteger)
    cd = plp.LpVariable.dicts("cd", indexs=D, lowBound=0, cat=plp.LpInteger)
    y = plp.LpVariable.dicts("y", indexs=(V, V, T), cat=plp.LpBinary)

    # The objective function
    prob = plp.LpProblem("Cmax_Pulse_Flow", plp.LpMinimize)
    prob += makespan, "Makespan"

    # Flow depart from origins
    for i in O:
        prob += (plp.lpSum([y[i][j][t] for t in T
                                       for j in D]) <= q[i]
        ), f"R1_Flow_depart_from_origin_{i}"

    # Every destination has to be cleaned at some point (1)
    for j in D:
        prob += (plp.lpSum([y[i][j][t] for t in T
                                       for i in V if i != j]) == 1
        ), f"R2_Enter_{j}"

    # Flow conservation constraints (2)
    for j in D:
        for t in T:
            prob += (
                plp.lpSum([y[i][j][t - c[i][j] - p[j]] for i in V if i != j
                                                       if t - c[i][j] - p[j]>= T[0]])
                == plp.lpSum([y[j][i][t] for i in V if i != j])
            ), f"R4_Flow_conservation_on_{j}_at_period_{t}"

    # Calculates the start time of every node
    for j in D:
        prob += (sd[j] == plp.lpSum([t*y[i][j][t] for t in T
                                                  for i in V if i != j])
        ), f"R5_Start_date_of_{j}"

    # Precedence constraints
    for i, j in G.precedencies:
        prob += sd[i] <= sd[j], f"R6_Start_cleaning_{i}_before_{j}"

    # Calculates the completion time of every node
    for j in D:
        prob += (cd[j] == plp.lpSum([(t + c[i][j]) * y[i][j][t] for t in T
                                                                for i in V if i != j])
                          + p[j]
        ), f"R7_Completion_date_of_{j}"

    # Calculates the makespan
    for j in D:
        prob += makespan >= cd[j], f"R8_Cmax_geq_C_{j}"

    return prob

'''
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
'''

@click.command()
@click.argument("instance-file")
@click.option("--time-limit",
              help="The maximum time limit for the execution (in seconds).")
@click.option("--log-path", help="File to write the execution log.")
@click.option("--sol-path", help="File to write the solution (in json).")
def run(instance_file, time_limit=None, log_path=None, sol_path=None):
    """Runs the model from command line.

       USAGE:

       python cmax_pulse_flow.py [OPTIONS] INSTANCE_FILE

       OPTIONS:

       --help\n
            show this message and exit
       
       --time-limit[=LIMIT]\n
            the maximum time limit for the execution (in seconds)

       --log-path[=LOG_PATH]\n
            where to write the execution log

       --sol-path[=SOL_PATH]\n
            where to write the solution (in Pulp json format)
    """
    G = loads(instance_file)
    prob = make_prob(G)
    prob.writeLP("CmaxPulseFlow.lp")

    # Solves the problem with CPLEX (assuming CPLEX is availible)
    solver = plp.getSolver('CPLEX_CMD', timeLimit=time_limit, logPath=log_path)
    prob.solve(solver)
    prob.roundSolution()

    # Exports the solution to a json file
    if sol_path:
        prob.to_json(sol_path, indent=2)

    # Prints variables with it's resolved optimum value
    print("")
    for v in prob.variables():
        if v.varValue and v.varValue > 0:
            print(v.name, "=", v.varValue)


if __name__ == "__main__":
    run()
