"""
    MILP gererator.

    Improvements:
        TODO: Load problem data from a file.
        TODO: The WT need to be identifiable by adding another index to x_i^t.
              So it's possible to know each one is been allocated.
"""

# constants
N = 6

# sets
V = tuple(range(0, N))
O = (0, 6)
D = (1, 2, 3, 4, 5)

# constants
r = {0: 0, 1: 0.8, 2: 0.3, 3: 0.4, 4: 0.2, 5: 0.75, 6: 0}
p = {0: 0, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0}
q = {0: 2, 6: 0}

# set T as an upperbound
T = sum(p.values())


def objective_function():
    """The objective function."""
    R = sum(r.values()) * T
    return "min " + " ".join(f"- {r[i]} z{i}.{t}" for i in D for t in range(1, T+1)) + f" + {R}"
                                                  


def eq_nb_wt(id=2):
    """Equations to ensure that the maximum number of WT is respected.
    """
    def _gen():
        for t in range(1, T+1):
            yield (f"c{id}.{t}: "
                   + " + ".join(f"x{i}.{t}" for i in D)
                   + f" <= {sum(q.values())}")

    return "\n".join(_gen())


def eq_repear(id=4):
    """Equations to ensure that every node gets the exacly amount of work it needs.
       Setting the z_i^t properly.
    """
    def _gen():
        for c, i in enumerate(D):
            for t in range(1, T+1):
                yield (f"c{id}.{t}.{c}: {p[i]} z{i}.{t}"
                       + "".join(f" - x{i}.{tt}" for tt in range(1, t+1))
                       + " <= 0")

    return "\n".join(_gen())


def eq_repeared(id=5):
    """Equations to ensure that all nodes are cleaned at the last period."""
    return f"c{id}: " + " + ".join(f"z{i}.{T}" for i in D) + f" = {len(D)}"


def vars():
    """Model variables."""
    
    def _gen():
        yield "\nbinaries\n"
        yield "\n".join(f"x{i}.{t}" for t in range(1, T+1)
                                    for i in D)
        yield ""
        yield "\n".join(f"z{i}.{t}" for t in range(1, T+1)
                                    for i in D)

    return "\n".join(_gen())


if __name__ == "__main__":
    print(objective_function(), end="\n\n")
    print("st", end="\n\n")
    print(eq_nb_wt(), end="\n\n")
    print(eq_repear(), end="\n\n")
    print(eq_repeared(), end="\n\n")
    print(vars(), end="\n\n")
    print("end")
