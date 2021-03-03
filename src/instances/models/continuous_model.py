"""
    MILP gererator.

    Improvements:
        TODO: Load problem data from a file.
        TODO: The WT need to be identifiable by adding a third index to x_ij.
              So it's possible to know each one is been allocated.
"""

# constants
N = 7
M = 999 # A big number

# sets
V = tuple(range(0, N))
O = (0, 6)
D = (1, 2, 3, 4, 5)

# constants
r = {0: 0, 1: 0.8, 2: 0.3, 3: 0.4, 4: 0.2, 5: 0.75, 6: 0}
p = {0: 0, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0}
q = {0: 1, 6: 1}


def objective_function():
    """The objective function."""
    return "min Cmax"
                                                  

def eq_wt_depart(id=2):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for i in O:
        yield (f"c{id}.{i}: "
                + " + ".join(f"x{i}.{j}" for j in D)
                + f" = {q[i]}")


def eq_wt_flow(id=3):
    """Equations to ensure the flow conservation.
    """
    for j in V:
        yield (f"c{id}.{j}: "
                + " + ".join(f"x{i}.{j}" for i in V)
                + "".join(f" - x{j}.{i}" for i in V)
                + " = 0")


def eq_clean(id=4):
    """Equations to ensure that every node in D is serviced.
    """
    for j in D:
        yield (f"c{id}.{j}: "
                + " + ".join(f"x{i}.{j}" for i in V)
                + " = 1")


def eq_completion_time(id=5):
    """Equations to calculate the completion time of the nodes.
    """
    for j in D:
        for i in V:
            yield (f"c{id}.{j}.{i}: C{i} - C{j} + {M} x{i}.{j} + {p[j]} x{i}.{j} <= {M}")


def eq_cmax(id=6):
    """Equations to calculate the Cmax."""
    for j in V:
        yield f"c{id}.{j}: Cmax - C{j} >= 0"


def vars():
    """Model variables."""
    yield "bounds\n"
    yield "Cmax >= 0\n"
    yield "\n".join(f"C{j} >= 0" for j in V)
    yield "\nbinaries\n"
    yield "\n".join(f"x{i}.{j}" for i in V
                                for j in V if i != j)
                            

if __name__ == "__main__":
    print(objective_function(), end="\n\n")
    print("st", end="\n\n")
    print("\n".join(eq_wt_depart()), end="\n\n")
    print("\n".join(eq_wt_flow()), end="\n\n")
    print("\n".join(eq_clean()), end="\n\n")
    print("\n".join(eq_completion_time()), end="\n\n")
    print("\n".join(eq_cmax()), end="\n\n")
    print("\n".join(vars()), end="\n\n")
    print("end")
