import argparse
from pathlib import Path

import pulp as plp
import networkx as nx

from optlis import load_instance, export_solution
from optlis.solvers import solver_parser


def model_1(instance, relaxation_threshold=0.0):
    """Implements the RCPSP model to minimize the overall risk."""
    # The set of tasks to process
    D = instance.tasks
    # The duration of each task
    p = instance.node_durations
    # The risk at each destination
    r = instance.node_risks
    # The number of teams at the depot
    K = sum(instance.node_resources)
    # The estimated amount of time periods to process all jobs (T is an upper bound)
    # indexed from 1 to T
    T = instance.time_periods

    # Creates the model's variables
    overall_risk = plp.LpVariable("overall_risk", lowBound=0, cat=plp.LpContinuous)
    x = plp.LpVariable.dicts("x", indices=(D, T), cat=plp.LpBinary)
    S = plp.LpVariable.dicts("S", indices=D, lowBound=0, cat=plp.LpInteger)
    C = plp.LpVariable.dicts("C", indices=D, lowBound=0, cat=plp.LpInteger)
    makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpInteger)

    # The objective function
    prob = plp.LpProblem("Overall_Risk", plp.LpMinimize)
    prob += overall_risk

    # Calculates the overall risk
    prob += overall_risk == plp.lpSum(r[i] * C[i] for i in D)

    # Resource constraints
    for t in T:
        prob += (plp.lpSum([x[i][tau] for i in D
                                      for tau in range(t - p[i] + 1, t+1)
                                      if tau >= 0])
                <= K,
                f"C1_Resource_constraint_at_period_{t}")

    # Every task has to be processed at some point
    for i in D:
        prob += plp.lpSum([x[i][t] for t in T]) == 1, f"C2_Process_task_{i}"

    # Precedence constraints
    for i, j in instance.precedence(d=relaxation_threshold):
        prob += S[i] <= S[j], f"C3_Prioritize_{i}_over_{j}"

    # Calculates the start time of tasks
    for i in D:
        prob += S[i] == plp.lpSum([t*x[i][t] for t in T]), f"C4_Start_{i}"

    # Calculates the completion times of tasks
    for i in D:
        prob += C[i] == plp.lpSum((t + p[i]) * x[i][t] for t in T), f"C5_Completion_{i}"

    # Calculates the project makespan
    for i in D:
        prob += makespan >= C[i], f"C6_Cmax_geq_C_{i}"

    return prob


def model_2(instance, relaxation_threshold=0.0):
    """Implements the RCPSP model with travel times to minimize the overall risk."""
    V = instance.nodes
    # The set of depots
    O = instance.depots
    # The set of tasks to process
    D = instance.tasks
    # The duration of each task
    p = instance.node_durations
    # The risk at each destination
    r = instance.node_risks
    # The sequence-dependent setup times
    s = instance.setup_times
    # The number of teams at each node (it should be one depot node)
    k = instance.node_resources
    # The estimated amount of time periods to process all jobs (T is an upper bound)
    # indexed from 1 to T
    T = instance.time_periods

    # Creates the model's variables
    makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpInteger)
    overall_risk = plp.LpVariable("overall_risk", lowBound=0, cat=plp.LpContinuous)
    S = plp.LpVariable.dicts("S", indices=D, lowBound=0, cat=plp.LpInteger)
    C = plp.LpVariable.dicts("C", indices=D, lowBound=0, cat=plp.LpInteger)
    y = plp.LpVariable.dicts("y", indices=(V, V, T), cat=plp.LpBinary)

    # The objective function
    prob = plp.LpProblem("Overall_Risk", plp.LpMinimize)
    prob += overall_risk

    # Calculates the overall risk
    prob += overall_risk == plp.lpSum(r[i] * C[i] for i in D)

    # Flow depart from depot
    for i in O:
        prob += (plp.lpSum(y[i][j][t] for t in T
                                      for j in D) <= k[i]
        ), f"C1_Flow_depart_from_origin_{i}"

    # Flow must enter every task
    for j in D:
        prob += (plp.lpSum(y[i][j][t] for t in T
                                      for i in V if i != j) == 1
        ), f"C2_Enter_{j}"

    # Flow must leave every task
    for j in D:
        prob += (plp.lpSum(y[j][i][t] for t in T
                                      for i in V if i != j) == 1
        ), f"C3_Leave_{j}"

    # Flow conservation constraints (allows idle times between consecutive tasks)
    for j in D:
        prob += (
            plp.lpSum(t * y[j][i][t] for i in V if i != j
                                     for t in T)
            - C[j] >= 0
        ), f"C4_Flow_conservation_{j}"

    # Calculates the start times of tasks
    for j in D:
        prob += (S[j] == plp.lpSum(t * y[i][j][t] for t in T
                                                  for i in V if i != j)
        ), f"C5_Start_{j}"

    # Precedence constraints
    for i, j in instance.precedence(d=relaxation_threshold):
        prob += S[i] <= S[j], f"C6_Start_{i}_before_{j}"

    # Calculates the completion times of tasks
    for j in D:
        prob += (
            C[j] == plp.lpSum((t + s[i][j]) * y[i][j][t] for t in T
                                                         for i in V if i != j)
                    + p[j]), f"C7_Completion_{j}"

    # Calculates the makespan
    for j in D:
        prob += makespan >= C[j], f"C8_Cmax_geq_C_{j}"

    return prob


def optimize(instance, make_model, relaxation_threshold=0.0,
             time_limit=None, log_path=None, sol_path=None):
    """Runs the model for an instance."""
    prob = make_model(instance, relaxation_threshold)

    # TODO: configure how the MILP are exported
    # prob.writeLP("OverallStatickRisk.lp")

    if time_limit is not None and time_limit <= 0:
        time_limit = None

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    solver = plp.getSolver(
        'CPLEX_PY',
        timeLimit=time_limit,
        logPath=log_path
    )

    status = prob.solve(solver)
    print("Status:", plp.LpStatus.get(prob.status))
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

    return prob.status, prob.variables()


def from_command_line(args):
    instance = load_instance(args["instance-path"],
                             args["travel_times"])

    # Chooses models 1 or 2 based on the use of travel times
    make_model = model_2 if args["travel_times"] else model_1

    optimize(instance,
             make_model,
             args["relaxation"],
             args["time_limit"],
             args["log_path"],
             args["sol_path"])


if __name__ == "__main__":
    from_command_line()
