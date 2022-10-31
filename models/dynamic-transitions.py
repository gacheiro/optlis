import argparse
import itertools
from pathlib import Path

import pulp as plp
import networkx as nx
import numpy as np

from optlis import load_instance, export_solution

# (test only) problem data
M = 999999
EPSILON = 0.01
T = (1, 2, 3, 4, 5)
PRODUCTS = (0, 1, 2)
RISK = (0, 1, 0.8, 0.6, 0.4)
V = {
    0: 0,
    1: 1,
    2: 1,
    3: 0,
    4: 0,
    5: 0,
}


def dr(p):
    return 0.05


def mr(p, m):
    if m == p + 1 and p != 0:
        return 0.2
    return 0


def make_lp(instance):
    """Implements the mixed integer linear model for the problem."""
    # Creates the model's variables
    w = plp.LpVariable.dicts(
        "w", indices=(PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    u = plp.LpVariable.dicts("u", indices=(T,), lowBound=0, cat=plp.LpContinuous)
    x = plp.LpVariable.dicts("x", indices=(PRODUCTS, T), lowBound=0, cat=plp.LpBinary)
    y = plp.LpVariable.dicts("y", indices=(T,), lowBound=0, cat=plp.LpBinary)
    r = plp.LpVariable.dicts(
        "r", indices=(PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    d = plp.LpVariable.dicts(
        "d", indices=(PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    z = plp.LpVariable.dicts("z", indices=(T,), lowBound=0, cat=plp.LpBinary)
    q = plp.LpVariable.dicts(
        "q", indices=(PRODUCTS, PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )

    lp = plp.LpProblem("MIN_DYN", plp.LpMinimize)
    # # (test) minimize global risk
    lp += plp.lpSum(RISK[p] * w[p][t] for (p, t) in itertools.product(PRODUCTS, T))
    # (test) dummy objf
    # lp += 1
    # lp += plp.lpSum(w[p][t] for (p, t) in itertools.product(PRODUCTS, T))

    # (test) sum up products' concentration
    for t in T:
        lp += u[t] == plp.lpSum(w[p][t] for p in PRODUCTS)
        # lp += u[t] >= 1 - EPSILON
        # lp += u[t] <= 1 + EPSILON

    # Set initial concentration
    for p in PRODUCTS:
        lp += w[p][1] == V[p]

    # Product metabolization
    for t in T[1:]:
        for p, s in itertools.product(PRODUCTS, PRODUCTS):
            if s == 0:
                continue
            lp += q[p][s][t] >= (w[p][t - 1] - d[p][t]) * mr(p, s) - M * (
                x[p][t] + y[t]
            )
            lp += q[p][s][t] <= (w[p][t - 1] - d[p][t]) * mr(p, s)

    # Product degradation
    for t in T[1:]:
        for p in PRODUCTS:
            lp += d[p][t] == w[p][t - 1] * dr(p)

    # Updates concentration values based on performed operations
    for t in T[1:]:
        for p in PRODUCTS:
            lp += w[p][t] == (
                w[p][t - 1]
                - d[p][t]
                + plp.lpSum(q[s][p][t] for s in PRODUCTS)
                - plp.lpSum(q[p][s][t] for s in PRODUCTS)
                - r[p][t]
            )

    # Neutralizing operation (w[0][t] <- w[p][t-1])
    for t in T[1:]:
        for p in PRODUCTS:
            # The linearization of:
            # lp += q[p][0][t] == w[p][t - 1] * (1 - dr(p)) * x[p][t]
            lp += q[p][0][t] >= 0
            lp += q[p][0][t] <= w[p][t - 1] * (1 - dr(p))
            lp += q[p][0][t] <= M * x[p][t]
            lp += q[p][0][t] >= w[p][t - 1] * (1 - dr(p)) + M * x[p][t] - M

    # Removal operation (w[p..][t] <- 0)
    for t in T[1:]:
        for p in PRODUCTS:
            # The linearization of:
            # lp += r[p][t] == w[p][t - 1] * y[t]
            lp += r[p][t] >= 0
            lp += r[p][t] <= w[p][t - 1] * (1 - dr(p))
            lp += r[p][t] <= M * y[t]
            lp += r[p][t] >= w[p][t - 1] * (1 - dr(p)) + M * y[t] - M

    # Can't perform remove and neutralize at the same time
    for t in T:
        lp += plp.lpSum(x[p][t] for p in PRODUCTS) + y[t] <= 1

    # (test only) hardcode on-site ops
    # lp += x[1][3] == 1
    lp += plp.lpSum(x[p][t] for p, t in itertools.product(PRODUCTS, T)) == 0

    # (test only) hardcode on-site ops
    # lp += y[4] == 1
    lp += plp.lpSum(y[t] for t in T) == 1

    # (test only) disable operations at time 1 and 2
    lp += plp.lpSum(x[p][1] + x[p][2] for p in PRODUCTS) + y[1] + y[2] == 0

    # (fix) can't neutralize product 0
    lp += plp.lpSum(x[0][t] for t in T) == 0

    return lp


def optimize(
    instance,
    time_limit=None,
    log_path=None,
    sol_path=None,
):
    """Runs the model for an instance."""
    prob = make_lp(instance)

    # TODO: configure how the MILP are exported
    prob.writeLP("DynamicRisk.lp")

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    solver = plp.getSolver("CPLEX_PY", timeLimit=time_limit, logPath=log_path)

    prob.solve(solver)
    prob.roundSolution()

    # Prints variables with it's resolved optimum value
    print("")
    try:
        print(f"objective_function = {prob.objective.value():.4f}")
    except TypeError:
        pass

    lhs_size = max(len(v.name) for v in prob.variables())
    for v in prob.variables():
        if v.varValue:
            print(f"{v.name.ljust(lhs_size)} = {v.varValue:.3f}")

    # TODO: only write solution with it exists!
    if sol_path:
        sol_path = Path(sol_path)
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        export_solution({v.name: v.varValue for v in prob.variables()}, "", sol_path)


def from_command_line():
    from optlis.solvers import solver_parser

    parser = argparse.ArgumentParser(parents=[solver_parser])
    parser.add_argument(
        "--time-limit",
        type=int,
        help="maximum time limit for the execution (in seconds)",
    )
    args = vars(parser.parse_args())

    instance = load_instance(args["instance-path"], args["setup_times"])

    optimize(
        instance,
        args["time_limit"],
        args["log_path"],
        args["sol_path"],
    )


if __name__ == "__main__":
    from_command_line()
