"""
    MILP gererator.

    Improvements:
        TODO: Load problem data from a file.
        TODO: The WT need to be identifiable by adding a third index to x_ij.
              So it's possible to know each one is been allocated.
"""
import click

from instances.inst import loads

# constants
G = None
N = 7

# sets
V = tuple(range(0, N))
O = (0, 6)
D = (1, 2, 3, 4, 5)

# constants
T = 0


def objective_function():
    """The objective function."""
    return "min Cmax"
                                                  

def eq_wt_depart(id=2):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for i in O:
        q_i = G.nodes[i]["q"]
        yield (f"c{id}.{i}: "
                + " + ".join(f"x{i}.{j}.{t}" for j in D
                                             for t in range(1, T+1))
                + f" <= {q_i}")


def eq_wt_depart2(id=3):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for t in range(2, T+1):
        for i in O:
            yield (f"c{id}.{i}.{t}: "
                    + " + ".join(f"x{i}.{j}.{t}" for j in V)
                    + f" = 0")


def eq_nb_wt(id=4):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for t in range(1, T+1):
        Q = sum(G.nodes[i]["q"] for i in G.nodes())
        yield (f"c{id}.{t}: "
                + " + ".join(f"x{i}.{j}.{tau}" for j in D
                                               for i in V
                                               for tau in range(t - G.nodes[j]["p"] + 1, t + 1)
                                               if tau > 0)
                + f" <= {Q}")


def eq_wt_flow(id=5):
    """Equations to ensure the flow conservation.
    """
    for j in D:
        p_j = G.nodes[j]["p"]
        for t in range(1, T+1-p_j):
            yield (f"c{id}.{j}.{t}: "
                    + " + ".join(f"x{i}.{j}.{t}" for i in V if i != j)
                    + "".join(f" - x{j}.{i}.{t + p_j}" for i in V if i != j)
                    + " = 0")


def eq_clean(id=6):
    """Equations to ensure that every node in D is serviced.
    """
    for j in D:
        yield (f"c{id}.{j}: "
                + " + ".join(f"x{i}.{j}.{t}" for i in V for t in range(1, T+1) if i != j)
                + " = 1")


def eq_clean2(id=7):
    """Equations to ensure that every node in D is serviced.
    """
    for j in D:
        yield (f"c{id}.{j}: "
                + " + ".join(f"x{j}.{i}.{t}" for i in V for t in range(1, T+1) if i != j)
                + " = 1")


def eq_completion_time(id=8):
    """Equations to calculate the completion time of the nodes.
    """
    for j in D:
        p_j = G.nodes[j]["p"]
        yield (f"c{id}.{j}: "
               + " + ".join(f"x{i}.{j}.{t}" for i in V for t in range(1, T+1) if i != j)
               + " ".join(f" + x{j}.{j}.{t}" for t in range(1, T+1))
               + f" = {p_j - 1}")


def eq_cj(id=9):
    """Equations to calculate the C_j."""
    for j in D:
        p_j = G.nodes[j]["p"]
        yield (f"c{id}.{j}: C{j} - "
               + " - ".join(f"{t} x{i}.{j}.{t}" for i in V for t in range(1, T+1) if i != j)
               + f" = {p_j}")


def eq_cmax(id=10):
    """Equations to calculate the C_max."""
    for j in V:
        yield f"c{id}.{j}: Cmax - C{j} >= 0"


def vars():
    """Model variables."""
    yield "bounds\n"
    yield "Cmax >= 0\n"
    yield "\n".join(f"C{j} >= 0" for j in V)
    yield "\nbinaries\n"
    yield " ".join(f"x{i}.{j}.{t}" for i in V
                                    for j in V
                                    for t in range(1, T+1)
                                    if i !=j)


@click.command()
@click.argument("path")
def generate_lp(path):
    global G, V, O, D, T
    G = loads(path)
    V = list(G.nodes())
    O = list(n for n in G.nodes() if G.nodes[n]["type"] == 0)
    D = list(n for n in G.nodes() if G.nodes[n]["type"] == 1)
    T = sum(G.nodes[n]["p"] for n in G.nodes())

    print(objective_function(), end="\n\n")
    print("st", end="\n\n")
    print("\n".join(eq_wt_depart()), end="\n\n")
    print("\n".join(eq_nb_wt()), end="\n\n")
#    print("\n".join(eq_wt_depart2()), end="\n\n")
    print("\n".join(eq_wt_flow()), end="\n\n")
    print("\n".join(eq_clean()), end="\n\n")
    print("\n".join(eq_clean2()), end="\n\n")
#    print("\n".join(eq_completion_time()), end="\n\n")
    print("\n".join(eq_cj()), end="\n\n")
    print("\n".join(eq_cmax()), end="\n\n")
    print("\n".join(vars()), end="\n\n")
    print("end")


if __name__ == "__main__":
    generate_lp()
