import argparse
import math
from typing import Dict, Any, Optional, Union
from itertools import product as set_product
from pathlib import Path

import pulp as plp

from optlis import export_solution
from optlis.dynamic import Instance, load_instance

# Problem constants
M = 999999
EPSILON = 0.01
CLEANING_SPEED = 0.15


def make_lp(instance: Instance):
    """Implements the mixed integer linear model for the problem."""

    # Problem data
    TASKS = instance.tasks
    DURATIONS = instance.durations
    RESOURCES = instance.resources
    T = instance.time_periods[1:]  # discard time unit 0
    PRODUCTS = instance.products
    RISK = instance.risk
    V = instance.initial_concentration

    dr = instance.degradation_rate
    mr = instance.metabolization_rate

    # Calculates the dynamic task duration
    def latest_start_date(i, t):
        """Returns the latest start time for i if i finishes exactly at time t."""
        for s in T:
            v = max(V[i][p] for p in PRODUCTS)
            tt = s
            while v > EPSILON:
                v -= CLEANING_SPEED
                tt += 1
            else:
                if tt == t:
                    return s
        return 1

    # Creates the model's variables
    global_risk = plp.LpVariable("global_risk", lowBound=0, cat=plp.LpContinuous)
    w = plp.LpVariable.dicts(
        "w", indices=(TASKS, PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    x = plp.LpVariable.dicts(
        "x", indices=(TASKS, PRODUCTS, T), lowBound=0, cat=plp.LpBinary
    )
    y = plp.LpVariable.dicts("y", indices=(TASKS, T), lowBound=0, cat=plp.LpBinary)
    n = plp.LpVariable.dicts("n", indices=(TASKS, T), lowBound=0, cat=plp.LpBinary)
    r = plp.LpVariable.dicts(
        "r", indices=(TASKS, PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    d = plp.LpVariable.dicts(
        "d", indices=(TASKS, PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    q = plp.LpVariable.dicts(
        "q", indices=(TASKS, PRODUCTS, PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    # makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpInteger)

    lp = plp.LpProblem("MIN_DYN", plp.LpMinimize)

    # Minimize global risk
    lp += global_risk

    # Minimize makespan
    # lp += makespan

    # Calculates solution's global risk
    lp += global_risk == plp.lpSum(
        RISK[p] * w[i][p][t] for i, p, t in set_product(TASKS, PRODUCTS, T)
    )

    # Sets initial concentration
    for i, p in set_product(TASKS, PRODUCTS):
        lp += w[i][p][1] == V[i][p]

    # Calculates products' metabolization
    for t, i in set_product(T[1:], TASKS):
        for p, s in set_product(PRODUCTS, PRODUCTS):
            if s == 0:
                continue
            lp += q[i][p][s][t] >= (w[i][p][t - 1] - d[i][p][t]) * mr(p, s) - M * (
                x[i][p][t] + y[i][t]
            )
            lp += q[i][p][s][t] <= (w[i][p][t - 1] - d[i][p][t]) * mr(p, s)

    # Calculates products' degradation
    for t, i, p in set_product(T[1:], TASKS, PRODUCTS):
        lp += d[i][p][t] == w[i][p][t - 1] * dr(p)

    # Updates concentration values based on performed operations
    for t, i, p in set_product(T[1:], TASKS, PRODUCTS):
        lp += w[i][p][t] == (
            w[i][p][t - 1]
            - d[i][p][t]
            + plp.lpSum(q[i][s][p][t] for s in PRODUCTS)
            - plp.lpSum(q[i][p][s][t] for s in PRODUCTS)
            - r[i][p][t]
        )

    # Neutralizing operation (w[0][t] <- w[p][t-1])
    # TODO: add neutralizing speed
    for t, i, p in set_product(T[1:], TASKS, PRODUCTS):

        # Checks wether a neutralizing operation is active
        time_window = range(t - DURATIONS[i] + 1, t + 1)
        is_active = plp.lpSum(
            x[i][p][tau] for p in PRODUCTS for tau in time_window if tau >= 1
        )

        # The linearization of:
        # lp += q[i][p][0][t] == (0.05 * w[i][p][t - 1]) * x[i][p][t] - d[i][p][t]
        lp += q[i][p][0][t] >= 0
        lp += q[i][p][0][t] <= 0.05 * w[i][p][t - 1] - d[i][p][t]
        lp += q[i][p][0][t] <= M * is_active
        lp += q[i][p][0][t] >= 0.05 * w[i][p][t - 1] - d[i][p][t] + M * is_active - M

    # Cleaning operation (w[p..][t] <- 0)
    for t, i, p in set_product(T[1:], TASKS, PRODUCTS):

        # Checks wether a cleaning operation is active
        time_window = range(latest_start_date(i, t) + 1, t + 1)
        is_active = plp.lpSum(y[i][tau] for tau in time_window if tau >= 1)

        # The linearization of:
        lp += r[i][p][t] <= CLEANING_SPEED * is_active
        # lp += r[i][p][t] >= 0
        # lp += r[i][p][t] <= 0.3 * w[i][p][t - 1] - d[i][p][t]
        # lp += r[i][p][t] <= M * is_active
        # lp += r[i][p][t] >= 0.3 * w[i][p][t - 1] - d[i][p][t] + M * is_active - M

    # Resource constraints (cleaning operation)
    for t in T:
        time_window = range(latest_start_date(i, t) + 1, t + 1)
        lp += (
            plp.lpSum(y[i][tau] for i in TASKS for tau in time_window if tau >= 1)
            <= RESOURCES["R"]
        )

    # Neutralizing operation active on task i at time t (n[i][t] == 1)
    for t, i in set_product(T, TASKS):
        # FIXME: neutralizing duration should be stated in the instance file
        time_window = range(t - DURATIONS[i] + 1, t + 1)
        lp += n[i][t] == plp.lpSum(
            x[i][p][tau] for p in PRODUCTS for tau in time_window if tau >= 1
        )

    # Resource constraints (neutralizing operation)
    for t in T:
        lp += plp.lpSum(n[i][t] for i in TASKS) <= RESOURCES["N"]

    # Each site is serviced once
    for i in TASKS:
        if any(V[i][p] > 0 for p in PRODUCTS):
            lp += plp.lpSum(y[i][t] for t in T) == 1
        else:
            lp += plp.lpSum(y[i][t] for t in T) == 0

    # # Operations cannot overlap
    # for t, i in set_product(T, TASKS):
    #     lp += n[i][t] + c[i][t] <= 1

    # (test only) hardcode on-site ops
    # lp += x[1][1][3] == 1
    # lp += x[1][1][3] == 1
    # lp += plp.lpSum(x[i][p][t] for i, p, t in set_product(TASKS, PRODUCTS, T)) == 0

    # (test only) hardcode on-site ops
    # lp += y[1][2] == 1
    # lp += plp.lpSum(y[i][t] for i, t in set_product(TASKS, T)) == 1

    # (test only) disable operations at t = 1
    for i in TASKS:
        lp += plp.lpSum(x[i][p][1] for p in PRODUCTS) + y[i][1] == 0

    # (fix) can't neutralize product 0
    for i in TASKS:
        lp += plp.lpSum(x[i][0][t] for t in T) == 0

    return lp


def optimize(
    instance: Instance,
    time_limit: Optional[int] = None,
    log_path: Optional[Union[str, Path]] = None,
    sol_path: Optional[Union[str, Path]] = None,
):
    """Runs the model for an instance."""
    prob = make_lp(instance)

    # TODO: configure how the MILP are exported
    # prob.writeLP("DynamicRisk.lp")

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    solver = plp.getSolver("CPLEX_PY", timeLimit=time_limit, logPath=log_path)

    prob.solve(solver)
    prob.roundSolution()

    # Prints variables with their optimized values
    print("")
    try:
        print(f"objective_function = {prob.objective.value():.4f}")
    except TypeError:
        pass

    lhs_size = max(len(v.name) for v in prob.variables())
    for v in prob.variables():
        if v.varValue:
            formatted_value = v.varValue if v.isInteger() else f"{v.varValue:.5f}"
            print(f"{v.name.ljust(lhs_size)} = {formatted_value}")

    # TODO: only write solution with it exists!
    if sol_path:
        sol_path = Path(sol_path)
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        export_solution({v.name: v.varValue for v in prob.variables()}, "", sol_path)


def from_command_line(args: Dict[str, Any]) -> None:
    instance = load_instance(args["instance-path"])

    optimize(
        instance,
        args["time_limit"],
        args["log_path"],
        args["sol_path"],
    )
