"""
    MILP gererator.

    A copy of the scheduling model, but with markov chains. The markov chains
    and the transition matrices are precalculated and the values are inserted
    in de model as constants.

    Improvements:
        TODO: Improve the objective function.
        TODO: Load problem data from a file.
        TODO: The WT need to be identifiable by adding another index to x_i^t.
              So it's possible to know each one is been allocated.
        TODO: Make the code legible.
"""

import numpy as np


# Constants
N = 7

# Initial state of the risk (low, medium, high)
R = {
    1: np.array([0, 0, 1]),
    2: np.array([1, 0, 0]),
    3: np.array([0, 0, 1]),
    4: np.array([1, 0, 0]),
    5: np.array([0, 1, 0]),
}

# The transition matrices. Defines 4 types of products
# Product A: stable product, doesn't change much between periods
A = np.array([[.9 , .07, .03],
              [.03, .9 , .02],
              [.05, .05, .9 ]])

# Product B: unstable product, the risk may get significantly worst
# between periods
B = np.array([[.5, .4 , .1 ],
              [.1, .6 , .3 ],
              [.0, .05, .95]])

# Product C: unstable product, the risk may vanish between periods
C = np.array([[.9, .05, .05],
              [.2, .75, .05],
              [.1, .3 , .6 ]])

# Product D: stable product, the risk stays the same betwwen periods
D = np.array([[1., .0, .0],
              [.0, 1., .0],
              [.0, .0, 1.]])

# The products in the nodes
P = {
    1: A,
    2: B,
    3: D,
    4: A,
    5: B,
}


# Sets
V = tuple(range(0, N))
O = (0, 6)
D = (1, 2, 3, 4, 5)

# Constants
p = {0: 0, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 0}
q = {0: 2, 6: 0}

# Set T as an upperbound
T = sum(p.values())


def objective_function():
    """The objective function."""
    A_sum = sum( (R[i] @ np.linalg.matrix_power(P[i], t))[2] for i in D for t in range(1, T+1) )

    return ("min "
            + " ".join(f"- { (R[i] @ np.linalg.matrix_power(P[i], t))[2] } z{i}.{t}" for i in D for t in range(1, T+1))
            + f" + {A_sum}")


def eq_nb_wt(id=2):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for t in range(1, T+1):
        yield (f"c{id}.{t}: "
                + " + ".join(f"x{i}.{t}" for i in D)
                + f" <= {sum(q.values())}")


def eq_repear(id=4):
    """Equations to ensure that every node gets the exacly amount of work it needs.
       Setting the z_i^t properly.
    """
    for c, i in enumerate(D):
        for t in range(1, T+1):
            yield (f"c{id}.{t}.{c}: {p[i]} z{i}.{t}"
                    + "".join(f" - x{i}.{tt}" for tt in range(1, t+1))
                    + " <= 0")


def eq_repeared(id=5):
    """Equations to ensure that all nodes are cleaned at the last period."""
    yield f"c{id}: " + " + ".join(f"z{i}.{T}" for i in D) + f" = {len(D)}"


def vars():
    """Model variables."""
    yield "binaries\n"
    yield "\n".join(f"x{i}.{t}" for t in range(1, T+1)
                                for i in D)
    yield ""
    yield "\n".join(f"z{i}.{t}" for t in range(1, T+1)
                                for i in D)


if __name__ == "__main__":
    print(objective_function(), end="\n\n")
    print("st", end="\n\n")
    print("\n".join(eq_nb_wt()), end="\n\n")
    print("\n".join(eq_repear()), end="\n\n")
    print("\n".join(eq_repeared()), end="\n\n")
    print("\n".join(vars()), end="\n\n")
    print("end")
