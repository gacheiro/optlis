import argparse
from pathlib import Path

import pulp as plp
import networkx as nx

from optlis import load_instance, export_solution
from optlis.solvers import solver_parser


def make_prob(G, relaxation_threshold=0.0):
    """Implements an integer programming model to minimize the overall risk."""
    V = G.nodes
    # The set of origins
    O = G.origins
    # The set of 'jobs' to process
    D = G.destinations
    # The duration of each job
    p = G.task_durations
    # The risk at each destination
    r = G.task_risks
    # The distance between every pair of nodes
    c = G.setup_times
    # The number of wt at each origin
    q = nx.get_node_attributes(G, "q")
    # The estimated amount of time periods to process all jobs (T is an upper bound)
    # indexed from 1 to T
    T = G.time_periods

    # Creates the model's variables
    makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpInteger)
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


def optimize(instance, relaxation_threshold=0.0,
             time_limit=None, log_path=None, sol_path=None):
    """Runs the model for an instance."""
    prob = make_prob(instance, relaxation_threshold)

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
                        "", sol_path)


def from_command_line():
    parser = argparse.ArgumentParser(parents=[solver_parser])
    parser.add_argument("--time-limit", type=int,
                        help="maximum time limit for the execution (in seconds)")
    args = vars(parser.parse_args())

    instance = load_instance(args["instance-path"],
                             args["setup_times"])
    optimize(instance,
             args["relaxation"],
             args["time_limit"],
             args["log_path"],
             args["sol_path"])


if __name__ == "__main__":
    from_command_line()
