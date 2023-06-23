from typing import Dict, Any, Optional, Union
from pathlib import Path

import pulp as plp

from optlis.shared import set_product, export_solution
from optlis.dynamic.problem_data import Instance, load_instance

# Problem constants
M = 999999


def make_lp(instance: Instance):
    """Implements the mixed integer linear model for the problem."""

    # Problem data
    TASKS = instance.tasks
    RESOURCES = instance.resources
    T = instance.time_units
    PRODUCTS = instance.products
    RISK = instance.products_risk
    V = instance.initial_concentration
    CLEANING_SPEED = instance.CLEANING_SPEED
    NEUTRALIZING_SPEED = instance.NEUTRALIZING_SPEED
    CLEANING_START_TIMES = instance.cleaning_start_times
    NEUTRALIZING_START_TIMES = instance.neutralizing_start_times

    # Defines aliases for some methods
    dr = lambda p: instance.degradation_rates[p]
    mr = lambda p, q: instance.metabolizing_rates[p][q]
    # nd = instance.neutralizing_duration

    # Creates the model's variables
    global_risk = plp.LpVariable("global_risk", lowBound=0, cat=plp.LpContinuous)
    w = plp.LpVariable.dicts(
        "w", indices=(TASKS, PRODUCTS, T), lowBound=0, cat=plp.LpContinuous
    )
    x = plp.LpVariable.dicts(
        "x", indices=(TASKS, PRODUCTS, T), lowBound=0, cat=plp.LpBinary
    )
    y = plp.LpVariable.dicts("y", indices=(TASKS, T), lowBound=0, cat=plp.LpBinary)
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

    # Calculates solution's global risk
    lp += global_risk == plp.lpSum(
        RISK[p] * w[i][p][t] for i, p, t in set_product(TASKS, PRODUCTS, T)
    )

    # Sets initial concentration
    for i, p in set_product(TASKS, PRODUCTS):
        lp += w[i][p][T[0]] == V(i, p)

    # Calculates products' metabolization
    for t, i in set_product(T[1:], TASKS):
        for p, s in set_product(PRODUCTS, PRODUCTS):
            if s == 0:
                continue
            # lp += q[i][p][s][t] >= (w[i][p][t - 1] - d[i][p][t]) * mr(p, s) - M * (
            #     x[i][p][t] + y[i][t]
            # )
            lp += q[i][p][s][t] == (w[i][p][t - 1] - d[i][p][t]) * mr(p, s)

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

    # Cleaning operation
    for i, p, t in set_product(TASKS, PRODUCTS, T):
        time_window = range(CLEANING_START_TIMES[i][t], t + 1)
        lp += r[i][p][t] <= CLEANING_SPEED * plp.lpSum(y[i][tau] for tau in time_window)

    # Resource constraints (cleaning operation)
    for t in T:
        time_window = range(CLEANING_START_TIMES[i][t], t + 1)
        lp += (
            plp.lpSum(y[i][tau] for i in TASKS for tau in time_window)
            <= RESOURCES["Qc"]
        )

    # Neutralizing operation
    for i, p, t in set_product(TASKS, PRODUCTS, T[1:]):

        # Checks wether a neutralizing operation is active
        time_window = range(NEUTRALIZING_START_TIMES[i][p][t], t + 1)
        is_active = plp.lpSum(x[i][p][tau] for tau in time_window)

        lp += q[i][p][0][t] >= 0
        lp += q[i][p][0][t] <= NEUTRALIZING_SPEED * w[i][p][t - 1] - d[i][p][t]
        lp += q[i][p][0][t] <= M * is_active
        lp += (
            q[i][p][0][t]
            >= NEUTRALIZING_SPEED * w[i][p][t - 1] - d[i][p][t] + M * is_active - M
        )

    # Resource constraints (neutralizing operation)
    for t in T:
        lp += (
            plp.lpSum(
                x[i][p][tau]
                for i in TASKS
                for p in PRODUCTS
                for tau in range(NEUTRALIZING_START_TIMES[i][p][t], t + 1)
            )
            <= RESOURCES["Qn"]
        )

    # Each site is cleaned no more than one time
    for i in TASKS:
        lp += plp.lpSum(y[i][t] for t in T) <= 1

        # Each product is neutralized no more than one time
        for p in PRODUCTS:
            lp += plp.lpSum(x[i][p][t] for t in T) <= 1

    # On-site operations cannot overlap
    for i, t in set_product(TASKS, T):

        cleaning = plp.lpSum(
            y[i][tau] for tau in range(CLEANING_START_TIMES[i][t], t + 1)
        )

        neutralizing = plp.lpSum(
            x[i][p][tau]
            for p in PRODUCTS
            for tau in range(NEUTRALIZING_START_TIMES[i][p][t], t + 1)
        )

        lp += cleaning + neutralizing <= 1

    # (test only) hardcode on-site ops
    # lp += x[1][1][1] == 1
    # lp += plp.lpSum(x[i][p][t] for i, p, t in set_product(TASKS, PRODUCTS, T)) == 0

    # (test only) hardcode on-site ops
    # lp += y[1][1] == 1
    # lp += r[1][1][1] == 0.15
    # lp += plp.lpSum(y[i][t] for i, t in set_product(TASKS, T)) == 0

    # (bug) avoid operations when there's no resource
    if RESOURCES["Qn"] == 0:
        lp += plp.lpSum(x[i][p][t] for i, p, t in set_product(TASKS, PRODUCTS, T)) == 0

    if RESOURCES["Qc"] == 0:
        lp += plp.lpSum(y[i][t] for i, t in set_product(TASKS, T)) == 0

    # (test only) disable operations at t = 0
    for i in TASKS:
        lp += plp.lpSum(x[i][p][0] for p in PRODUCTS) + y[i][0] == 0

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
    prob.writeLP("DynamicRisk.lp")

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

    # TODO: only write solution if it exists!
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
