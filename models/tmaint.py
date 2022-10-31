import math
from functools import lru_cache

import numpy as np
import pulp as plp

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


def decision_cost(decision):
    if decision_names[decision] == "maintenance":
        return -500
    elif decision_names[decision] == "fix":
        return -2000
    return 0


def state_reward(state):
    if state_names[state] == "good":
        return 1500
    elif state_names[state] == "average":
        return 1100
    elif state_names[state] == "bad":
        return 700
    return 0


@lru_cache
def r(t, state):
    if t == 100:
        return 0
    rs = []
    for d in possible_decisions[state_names[state]]:
        expected = 0
        for s in states:
            expected += (state_reward(state) + decision_cost(d) + β*r(t+1, s))*p[state][s]
        rs.append(expected)
    return max(rs)


def make_lp():
    """Implements the truck maintenance non-stationary model."""

    expected_profit = plp.LpVariable.dicts("💵", cat=plp.LpContinuous,
                                           indices=states)

    # The objective function
    lp = plp.LpProblem("Markov_Manut", plp.LpMinimize)
    lp += plp.lpSum(expected_profit[i] for i in states), "Profit"

    for i in states:
        lp += expected_profit[i] >= r(0, states[i])

    # print("Interesting: r(i, t) ~= r(i, t+t')")
    # print(r(0, states[0]), r(95, states[0]))
    # print(r(0, states[1]))
    # print(r(0, states[2]))
    # print(r(0, states[3]), r(98, states[3]))
    # print(r(0, states[4]))

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

    # Prints variables with it's resolved optimum value
    print("")
    for v in lp.variables():
        print(f"{v.name} = {v.varValue:.3f}")
    print("\nOptimal policy: use the truck until it breaks!")


if __name__ == "__main__":
    run_instance()
