import math
from functools import lru_cache

import numpy as np
import pulp as plp

tasks = [1]
states = [0, 1, 2, 3, 4]
state_names = ["good", "average", "bad", "broken", "being fixed"]
decisions = [0, 1, 2]
decision_names = ["do nothing", "maintenance", "fix"]
β = 0.035
p = np.array([(0.8, 0.15, 0.04, 0.01, 0.),    # good
              (0. , 0.8 , 0.15, 0.05, 0.),    # average
              (0. , 0.  , 0.8 , 0.2 , 0.),    # bad
              (0. , 0.  , 0.  , 0.  , 1.),    # broken
              (1. , 0.  , 0.  , 0.  , 0.)])   # been fixed

possible_decisions = {
    "good": [0],
    "average": [0, 1],
    "bad": [0, 1],
    "broken": [2],
    "being fixed": [0],
}
T = 10
H = list(range(T))


def expected(state, decision):
    """Returns the expect profit for state `s`, assuming that you take a `decision`."""
    if decision_names[decision] == "maintenance":
        return -500
    elif decision_names[decision] == "fix":
        return -2000

    if state_names[state] == "good":
        return 1500
    elif state_names[state] == "average":
        return 1100
    elif state_names[state] == "bad":
        return 700
    elif state_names[state] == "broken":
        return 0 # -2000 <======
    elif state_names[state] == "being fixed":
        return 0

    raise ValueError(f"state = {state} decision = {decision}")


@lru_cache
def r(i, t, state):
    """Returns the expected reward."""
    if t == T:
        return 0
    rs = []
    # for d in decisions:
    for d in possible_decisions[state_names[state]]: # this forces the truck to be `fixed`
        next_e = 0
        for s in states:
            next_e += β*r(i, t+1, s)*p[state][s]
        rs.append(expected(state, d) + next_e)
    return max(rs)


def make_lp():
    """Implements the truck maintenance non-stationary model."""
    global z
    profit = plp.LpVariable("profit", cat=plp.LpContinuous)
    z = plp.LpVariable.dicts("z", cat=plp.LpBinary, indices=(tasks, H))
    S = plp.LpVariable.dicts("S", indices=tasks, lowBound=0, cat=plp.LpInteger)
    returns = plp.LpVariable.dicts("💵", cat=plp.LpContinuous,
                                   indices=(tasks, states))

    # The objective function
    lp = plp.LpProblem("Markov_Manut", plp.LpMinimize)
    lp += profit, "Profit"
    lp += profit >= plp.lpSum(returns[i][s] for i in tasks for s in states)

    for i in tasks:
        for t in H:
            for s in states:
                lp += returns[i][s] >= r(i, t, s)*(1 - z[i][t])

    # Apparently, these constraints are required for the `step` to work
    for i in tasks:
        lp += z[i][0] == 0

    # Resource constraints
    for t in H:
        lp += plp.lpSum((z[i][t] - z[i][max(t-5, 0)]) for i in tasks) <= 1

    # Step contraints
    for i in tasks:
        for t in H:
            lp += (z[i][t] - z[i][max(t-1, 0)]) >= 0

    # Calculates the start time of tasks
    for i in tasks:
        lp += S[i] >= plp.lpSum(t*(z[i][t] - z[i][max(t-1, 0)]) for t in H)

    # import pdb; pdb.set_trace()
    return lp


def run_instance():
    """Runs the model for an instance."""
    lp = make_lp()
    lp.writeLP("rTruckMan.lp")

    solver = plp.getSolver(
        "CPLEX_PY",
    )

    lp.solve(solver)
    lp.roundSolution()

    print("")
    for v in lp.variables():
        if v.varValue is not None:
            print(f"{v.name} = {v.varValue:.3f}")


if __name__ == "__main__":
    run_instance()
