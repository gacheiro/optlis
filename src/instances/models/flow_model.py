"""
    MILP gererator.

    Limitations:
        BUG: The model keeps sending WT to perform remaining work even if they are not required.
        BUG: The flow doesn't take a 'shortest path' strategy. It'll usually take a longer route
             than what is needed.
        BUG: Using the OF (1 - r_i)x_i^t the z_i^t is only set to 1 at the last period.

    Improvements:
        TODO: Load problem data from a file.
        TODO: Currently, the WT flow is reset at every period. It would be more appropriated
              if the flow could be sent from destinations as well, when the cleaning is done
              and the worktroop is free to move to other nodes.
"""

N = 7
E = 10

# sets
V = tuple(range(0, N))
A = (
    (0, 1), (0, 2),
    (1, 0), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 3), (2, 5),
    (3, 1), (3, 2), (3, 4), (3, 6),
    (4, 3), (4, 5), (4, 6),
    (5, 2), (5, 4),
    (6, 3), (6, 4),
)
O = (0, 6)
D = (1, 2, 3, 4, 5)

# constants
r = {0: 0, 1: 0.8, 2: 0.3, 3: 0.4, 4: 0.2, 5: 0.75, 6: 0}
p = {0: 0, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0}
q = {0: 1, 6: 1}

T = sum(p.values())


def objective_function():
    """The objective function."""
    return ("min "
            + " + ".join(f"{1 - r[i]} x{i}.{t}" for i in D for t in range(1, T+1)))


def eq_nb_wt(id=2):
    """Equations to ensure that the maximum number of WT is respected, from
       the origins.
    """
    for c, i in enumerate(O):
        arc_from_origins = [a for a in A if a[0] == i]
        for t in range(1, T+1):
            yield (f"c{id}.{t}.{c}: "
                    + " + ".join(f"y{i}.{j}.{t}" for i, j in arc_from_origins)
                    + f" <= {q[i]}")


def eq_wt_flow(id=3):
    """Equations to ensure the flow conservation of WT."""
    for c, j in enumerate(D):
        incoming = [a for a in A if a[1] == j]
        outgoing = [a for a in A if a[0] == j]
        for t in range(1, T+1):
            yield (f"c{id}.{t}.{c}: "
                    + " + ".join(f"y{i}.{j}.{t}" for i, j in incoming)
                    + "".join(f" - y{j}.{i}.{t}" for j, i in outgoing)
                    + f" - x{j}.{t} = 0")


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


"""
def c7():

    def _gen():
        for c, i in enumerate(D):
            outgoing = [a for a in A if a[0] == i]
            for t in range(1, T - p[i] + 1):
                yield (f"c7.{t}.{c}: "
                       + " + ".join(f"y{i}.{j}.{t + p[i]}" for i, j in outgoing)
                       + f" - x{i}.{t} >= 0")

    return "\n".join(_gen())
"""


def vars():
    """Model variables."""    
    yield "bounds\n"
    yield "\n".join(f"y{i}.{j}.{t} >= 0" for t in range(1, T+1)
                                            for i, j in A)
    yield "\nbinaries\n"
    yield "\n".join(f"x{i}.{t}" for t in range(1, T+1)
                                for i in D)
    yield ""
    yield "\n".join(f"z{i}.{t}" for t in range(1, T+1)
                                for i in D)
    yield "\ngeneral\n"
    yield "\n".join(f"y{i}.{j}.{t}" for t in range(1, T+1)
                                    for i, j in A)


if __name__ == "__main__":
    print(objective_function(), end="\n\n")
    print("st", end="\n\n")
    print("\n".join(eq_nb_wt()), end="\n\n")
    print("\n".join(eq_wt_flow()), end="\n\n")
    print("\n".join(eq_repear()), end="\n\n")
    print("\n".join(eq_repeared()), end="\n\n")
    print("\n".join(vars()), end="\n\n")
    print("end")
