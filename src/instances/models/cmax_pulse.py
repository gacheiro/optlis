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

# sets
V = []
O = []
D = []


def objective_function():
    """The objective function."""
    return "min Cmax"


def eq_nb_wt(id=2):
    """Equations to ensure that the maximum number of WT is respected.
    """
    for t in range(1, T+1):
        Q = sum(G.nodes[i]["q"] for i in G.nodes())
        yield (f"c{id}.{t}: "
                + " + ".join(f"x{i}.{tau}" for i in D
                                           for tau in range(t - G.nodes[i]["p"] + 1, t + 1)
                                           if tau >= 1)
                + f" <= {Q}")


def eq_clean(id=3):
    """Equations to ensure that every node in D is serviced.
    """
    for i in D:
        yield (f"c{id}.{i}: "
                + " + ".join(f"x{i}.{t}" for t in range(1, T+1))
                + " = 1")


def eq_start_time(id=4):
    """Equations to calculate the start time of the jobs.
    """
    for i in D:
        p_i = G.nodes[i]["p"]
        yield (f"c{id}.{i}: S{i} - "
               + " - ".join(f"{t} x{i}.{t}" for t in range(1, T+1)) + f" = 0")


def eq_completion_time(id=5):
    """Equations to calculate the completion time of the nodes.
    """
    for i in D:
        p_i = G.nodes[i]["p"]
        yield (f"c{id}.{i}: C{i} - S{i} = {p_i}")


def eq_cmax(id=6):
    """Equations to calculate the Cmax."""
    for j in V:
        yield f"c{id}.{j}: Cmax - C{j} >= 0"


def vars():
    """Model variables."""
    yield "bounds\n"
    yield "Cmax >= 0\n"
    yield "\n".join(f"S{j} >= 0" for j in D)
    yield "\n".join(f"C{j} >= 0" for j in D)
    yield "\ngeneral\n"
    yield "Cmax\n"
    yield "\n".join(f"S{j}" for j in D)
    yield "\n".join(f"C{j}" for j in D)
    yield "\nbinaries\n"
    yield "\n".join(f"x{i}.{t}" for i in D
                                for t in range(1, T+1))


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
    print("\n".join(eq_nb_wt()), end="\n\n")
    print("\n".join(eq_clean()), end="\n\n")
    print("\n".join(eq_start_time()), end="\n\n")
    print("\n".join(eq_completion_time()), end="\n\n")
    print("\n".join(eq_cmax()), end="\n\n")
    print("\n".join(vars()), end="\n\n")
    print("end")


if __name__ == "__main__":
    generate_lp()
