from math import isclose
from itertools import product as prod

import matplotlib.pyplot as plt

from optlis.shared import import_solution
from optlis.dynamic import load_instance


MAKESPAN = 100


def _risk_at_time(inst, sol, t):
    tasks = inst.tasks
    risk = inst.products_risk
    products = inst.products
    return sum(sol.get(f"w_{i}_{p}_{t}", 0) * risk[p] for i, p in prod(tasks, products))


def _plot_risk_by_time(ax, instance, solution, alpha=0.8, print_data=False):
    """Plot a graph for the risk vs. time."""
    makespan = solution.get("makespan", MAKESPAN)
    x = list(range(1, makespan + 1))
    y = []
    for t in range(1, makespan + 1):
        y.append(_risk_at_time(instance, solution, t))

    ax.fill_between(x, y, alpha=alpha, label="Global risk")

    ax.legend(loc="upper right")
    ax.set(xlabel="time", ylabel="risk")
    ax.set_title("Risk over time")

    if print_data:
        print(" ".join(f"({_x}, {_y:.2f})" for _x, _y in zip(x, y)))


def _concentration_at_time(inst, sol, p, t):
    return sum(sol.get(f"w_{i}_{p}_{t}", 0) for i in inst.tasks)


def _plot_concentration_by_time(ax, instance, solution, alpha=0.8):
    """Plot a graph for products' concentration vs. time."""
    makespan = solution.get("makespan", MAKESPAN)
    products = instance.products
    x = list(range(1, makespan + 1))

    for p in products:
        y = []
        for t in range(1, makespan + 1):
            y.append(_concentration_at_time(instance, solution, p, t))
        ax.fill_between(x, y, alpha=alpha, label=f"Product {p}")

    ax.set_title("Concentration over time")
    ax.set(xlabel="time", ylabel="concentration")
    ax.legend(loc="upper right")


def plot_graphs(instance_path, sol_path):
    instance = load_instance(instance_path)
    solution = import_solution(sol_path)

    print(f"Instance {instance_path} ", end="")
    print(f"(global risk = {solution.get('global_risk') : .5f}, ", end="")
    print(f"makespan = {solution.get('makespan')})")

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    _plot_risk_by_time(axs[0], instance, solution, True)
    _plot_concentration_by_time(axs[1], instance, solution)

    plt.show()
