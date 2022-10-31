import argparse
from functools import partial
from pathlib import Path

import pulp as plp
import networkx as nx
import numpy as np

from optlis import load_instance, export_solution

# Hardcoded risk states
r = np.array([[.7],
              [.5],
              [.1]])

# Hardcoded products
stable = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

decreasing = np.array([[.8,  .1,  .1],
                       [ 0, .98, .02],
                       [ 0,   0,   1]])

increasing = np.array([[  0,  1,   0],
                       [  0,  1,   0],
                       [.02,  0, .98]])


def get_risk(a, t):
    """Returns the probabilistic risk vector for a transition matrix at time t.

       Note: see https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation
    """
    at = np.linalg.matrix_power(a, t)
    return np.matmul(at, r)


def make_prob(instance, relaxation_threshold=0.0, no_graph=False):
    """Implements the mixed integer linear model for the problem."""
    # The set of tasks to process
    D = instance.tasks
    # The duration of each task
    p = instance.task_durations
    # The number of teams at each depot
    K = sum(nx.get_node_attributes(instance, "q").values())
    # The estimated amount of time periods to process all jobs (T is an upper bound)
    # indexed from 1 to T
    T = instance.time_periods

    # Creates the model's variables
    overall_risk = plp.LpVariable("overall_risk", lowBound=0, cat=plp.LpContinuous)
    S = plp.LpVariable.dicts("S", indices=D, lowBound=0, cat=plp.LpInteger)
    z = plp.LpVariable.dicts("z", indices=(D, T), lowBound=0, cat=plp.LpBinary)

    # Supposes every node is affected by the same product
    r = partial(get_risk, stable)

    # The objective function
    prob = plp.LpProblem("Markov_Min_Overall_Risk", plp.LpMinimize)
    prob += overall_risk

    # This is equivalent to r[i] * S[i], z[i][0] is required to be 0 (see below)
    prob += overall_risk == \
        plp.lpSum(np.max(r(t))*(1 - z[i][t]) for i in D for t in T[1:])

    # Apparently, these constraints are required for the `step` to work
    for i in D:
        prob += z[i][0] == 0

    # Resource constraints
    for t in T:
        prob += plp.lpSum((z[i][t] - z[i][max(t-p[i], 0)]) for i in D) <= K

    # Step contraints
    for i in D:
        for t in T:
            prob += z[i][t] - z[i][max(t-1, 0)] >= 0

    # Calculates the start time of tasks
    for i in D:
        prob += S[i] >= plp.lpSum(t*(z[i][t] - z[i][max(t-1, 0)]) for t in T)

    return prob


def optimize(G, relaxation_threshold=0.0, no_graph=False,
             time_limit=None, log_path=None, sol_path=None):
    """Runs the model for an instance."""
    prob = make_prob(G, relaxation_threshold, no_graph)

    # TODO: configure how the MILP are exported
    prob.writeLP("MarkovOverallRisk.lp")

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    solver = plp.getSolver(
        'CPLEX_PY',
        timeLimit=time_limit,
        logPath=log_path
    )

    prob.solve(solver)
    prob.roundSolution()

    # Prints variables with it's resolved optimum value
    print("")
    print(f"overall_risk = {prob.objective.value():.4f}")
    for v in prob.variables():
        if v.name.startswith("S"):
            print(v.name, "=", v.varValue)

    # TODO: only write solution with it exists!
    if sol_path:
        sol_path = Path(sol_path)
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        export_solution({v.name: v.varValue for v in prob.variables()},
                        instance_path, sol_path)


def from_command_line():
    from optlis.solvers import solver_parser
    parser = argparse.ArgumentParser(parents=[solver_parser])
    parser.add_argument("--time-limit", type=int,
                        help="maximum time limit for the execution (in seconds)")
    args = vars(parser.parse_args())

    instance = load_instance(args["instance-path"],
                             args["setup_times"])

    optimize(instance,
             make_prob,
             args["relaxation"],
             args["time_limit"],
             args["log_path"],
             args["sol_path"])


if __name__ == "__main__":
    from_command_line()
