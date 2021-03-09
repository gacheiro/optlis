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
M = 999 # A big number

# sets
V = []
O = []
D = []


def objective_function():
    """The objective function."""
    return "min Cmax"


def eq_wt_depart(id=2):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for i in O:
        q_i = G.nodes[i]["q"]
        yield (f"c{id}.{i}: "
                + " + ".join(f"x{i}.{j}" for j in D)
                + f" = {q_i}")


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
                + " + ".join(f"x{i}.{j}" for i in V if i != j)
                + " = 1")


def eq_completion_time(id=5):
    """Equations to calculate the completion time of the nodes.
    """
    for j in D:
        for i in V:
            p_j = G.nodes[j]["p"]
            yield (f"c{id}.{j}.{i}: C{i} - C{j} + {M + p_j} x{i}.{j} <= {M}")


def eq_cmax(id=6):
    """Equations to calculate the Cmax."""
    for j in V:
        yield f"c{id}.{j}: Cmax - C{j} >= 0"


def vars():
    """Model variables."""
    yield "bounds\n"
    yield "Cmax >= 0\n"
    yield "\n".join(f"C{j} >= 0" for j in V)
    yield "\ngeneral\n"
    yield "Cmax\n"
    yield "\n".join(f"C{j}" for j in V)
    yield "\nbinaries\n"
    yield "\n".join(f"x{i}.{j}" for i in V
                                for j in V if i != j)


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
    print("\n".join(eq_wt_flow()), end="\n\n")
    print("\n".join(eq_clean()), end="\n\n")
    print("\n".join(eq_completion_time()), end="\n\n")
    print("\n".join(eq_cmax()), end="\n\n")
    print("\n".join(vars()), end="\n\n")
    print("end")


if __name__ == "__main__":
    generate_lp()
