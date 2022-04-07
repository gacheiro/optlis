from pathlib import Path

import pulp as plp
import networkx as nx
import click

from instances import load_instance, export_solution


def make_prob(G, relaxation_threshold=0.0, no_setup_times=False):
    """Implements an integer programming model to minimize the overall risk."""
    V = G.nodes
    # The set of origins
    O = G.origins
    # The set of 'jobs' to process
    D = G.destinations
    # The number of wt at each origin
    q = nx.get_node_attributes(G, "q")
    # The duration of each job
    p = nx.get_node_attributes(G, "p")
    # The risk at each destination
    r = nx.get_node_attributes(G, "r")
    # The distance between every pair of nodes
    c = dict(nx.shortest_path_length(G))
    # Does not take in to consideration the cost to traverse
    # through the graph
    if no_setup_times:
        for i in V:
            for j in V:
                c[i][j] = 0
    # The estimated amount of time periods to process all jobs (T is an upper bound)
    # indexed from 1 to T
    T = G.time_periods

    # Creates the model's variables
    makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpContinuous)
    overall_risk = plp.LpVariable("overall_risk", lowBound=0, cat=plp.LpContinuous)
    sd = plp.LpVariable.dicts("sd", indexs=D, lowBound=0, cat=plp.LpInteger)
    cd = plp.LpVariable.dicts("cd", indexs=D, lowBound=0, cat=plp.LpInteger)
    y = plp.LpVariable.dicts("y", indexs=(V, V, T), cat=plp.LpBinary)

    # The objective function
    prob = plp.LpProblem("Overall_Risk", plp.LpMinimize)
    prob += overall_risk

    # Calculates the overall risk
    prob += overall_risk == plp.lpSum(r[i] * cd[i] for i in D)

    # Calculates the makespan
    for j in D:
        prob += makespan >= cd[j]

    # Flow depart from origins
    for i in O:
        prob += (plp.lpSum(y[i][j][t] for t in T
                                      for j in D) <= q[i]
        ), f"R1_Flow_depart_from_origin_{i}"

    # Flow must enter every job
    for j in D:
        prob += (plp.lpSum(y[i][j][t] for t in T
                                      for i in V if i != j) == 1
        ), f"R2_Enter_{j}"

    # Flow must leave every job
    for j in D:
        prob += (plp.lpSum(y[j][i][t] for t in T
                                      for i in V if i != j) == 1
        ), f"R3_Leave_{j}"

    # Flow conservation constraints (allows idle times between jobs)
    for j in D:
        prob += (
            plp.lpSum(t * y[j][i][t] for i in V if i != j
                                     for t in T)
            - cd[j] >= 0
        ), f"R4_Flow_conservation_{j}"

    # Calculates the start time of every node
    for j in D:
        prob += (sd[j] == plp.lpSum(t * y[i][j][t] for t in T
                                                   for i in V if i != j)
        ), f"R5_Start_of_{j}"

    # Precedence constraints
    for i, j in G.dag(p=relaxation_threshold):
        prob += sd[i] <= sd[j], f"R6_Start_{i}_before_{j}"

    # Calculates the completion time of every node
    for j in D:
        prob += (
            cd[j] == plp.lpSum((t + c[i][j]) * y[i][j][t] for t in T
                                                          for i in V if i != j)
                    + p[j]), f"R7_Completion_of_{j}"

    return prob


def run_instance(instance_path="", relaxation_threshold=0.0, no_setup_times=False,
                 time_limit=None, log_path=None, sol_path=None):
    """Runs the model for an instance."""
    G = load_instance(instance_path)
    prob = make_prob(G, relaxation_threshold, no_setup_times)

    # TODO: configure how the MILP are exported
    prob.writeLP("OverallStatickRisk.lp")

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    cplex_args = dict(
        timeLimit=time_limit,
        logPath=log_path
    )

    # Solves the problem with CPLEX (assumes CPLEX is availible)
    try:
        # Tries to disable PulP fixed mip re-optimization
        # This needs to use fork https://github.com/thiagojobson/pulp
        solver = plp.getSolver(
            'CPLEX_CMD',
            **cplex_args,
            reoptimizeFixedMip=False,
        )
    except TypeError:
        # In case the `reoptimizeFixedMip` flag is not supported (default PulP)
        solver = plp.getSolver(
            'CPLEX_CMD',
            **cplex_args,
        )

    prob.solve(solver)
    prob.roundSolution()

    # Prints variables with it's resolved optimum value
    print("")
    for v in prob.variables():
        if v.varValue:
            print(v.name, "=", v.varValue)

    # TODO: only write solution with it exists!
    if sol_path:
        sol_path = Path(sol_path)
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        export_solution({v.name: v.varValue for v in prob.variables()},
                        instance_path, sol_path)


@click.command()
@click.argument("instance-path")
@click.option("--relaxation", type=float, default=0.0,
              help="Relaxation threshold for the priority rules.")
@click.option("--no-setup-times", is_flag=True, default=False,
              help="Ignore the cost of traversing through the graph.")
@click.option("--time-limit", type=int,
              help="The maximum time limit for the execution (in seconds).")
@click.option("--log-path", type=click.Path(),
              help="Path to write the execution log.")
@click.option("--sol-path", type=click.Path(),
              help="Path to write the solution.")
def command_line(instance_path, relaxation=0.0, no_setup_times=False, time_limit=None,
                 log_path=None, sol_path=None):
    """Runs the model from command line.

       USAGE:

       python cplex.py [OPTIONS] INSTANCE_PATH

       OPTIONS:

       --help\n
            show this message and exit\n
       --relaxation[=THRESHOLD]\n
            the threshold relaxation for the priority rules (in range [0, 1]).\n
            Default is 0 (strict priority rule).\n
       --no-setup-times\n
            flag to ignore the cost of traversing through the graph\n
       --time-limit[=LIMIT]\n
            the maximum time limit for the execution (in seconds)\n
       --log-path[=LOG_PATH]\n
            path to write the execution log\n
       --sol-path[=SOL_PATH]\n
            path to write the solution variables\n
    """
    run_instance(instance_path, relaxation, no_setup_times, time_limit,
                 log_path, sol_path)


if __name__ == "__main__":
    command_line()
