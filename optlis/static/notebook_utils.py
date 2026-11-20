import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

from optlis.shared import import_solution
from optlis.static.problem_data import load_instance


def y_axis(G, sol=None):
    """Generates the y values (the sum of the risk of not cleaned sites)
    for plotting a solution."""
    sol = sol or {}
    dest = G.tasks
    r = nx.get_node_attributes(G, "r")
    cd = {i: _completion_date(sol, i) for i in dest}
    if None in cd.values():
        print(cd.values())
        raise ValueError("some of the completion dates are None")
    makespan = int(sol.get("makespan", 0))
    for t in range(1, makespan + 1):
        # Calculates the accumulated risk of uncleaned sites in the period.
        # Still consider r[i] when cd[i] == t
        yield sum(r[i] for i in dest if cd[i] >= t)


def _completion_date(sol, i):
    # Compatibility between two types of solution formats
    return sol.get(f"cd_{i}") or sol.get(f"C_{i}")


def _start_date(sol, i):
    # Compatibility between two types of solution formats
    return sol.get(f"sd_{i}") or sol.get(f"S_{i}", 0)


def plot_overall_risk(
    instance_path, sol_paths=(), alpha=0.8, labels=(), print_data=False
):
    """Plot solutions sobreposed over a graph of risk per time period."""
    x, y = [], []
    G = load_instance(instance_path)
    print(f"Instance {instance_path}")
    _fig, ax = plt.subplots()
    for i, sol_path in enumerate(sol_paths):
        try:
            sol = import_solution(sol_path)
        except FileNotFoundError:
            continue
        makespan = int(sol.get("makespan", 0))
        x = list(range(1, makespan + 1))
        y = list(y_axis(G, sol))
        try:
            label = labels[i]
        except IndexError:
            label = f"sol {i}"
        print(f"\n{label} makespan: {sol.get('makespan')}, area: {sum(y):.2f}")
        if print_data:
            print(" ".join(f"({_x}, {_y:.2f})" for _x, _y in zip(x, y, strict=True)))
        ax.fill_between(x, y, alpha=alpha, label=label)
    ax.set(xlabel="time", ylabel="accumulated risk")
    ax.legend(loc="upper right")
    if print_data:
        print(" ".join(f"({_x}, {_y:.2f})" for _x, _y in zip(x, y, strict=True)))
    plt.show()


def makespan(sol):
    """Returns the makespan of a solution."""
    return sol["makespan"]


def overall_risk(G, sol=None):
    """Returns the overall risk of a solution."""
    sol = sol or {}
    acc_risk = sum(y_axis(G, sol))
    # Checks for the accumulated risk in the solution.
    # If it's not defined, we ignore and return the computed value
    assert math.isclose(sol.get("overall_risk", acc_risk), acc_risk), (
        f"accumulated risks differ {acc_risk} != {sol['overall_risk']}"
    )
    return acc_risk


def plot_gantt_diagram(instance_path, sol_path, figsize=(8, 8)):
    """Plots the grantt diagram for a given instance and solution."""
    inst, sol = (load_instance(instance_path), import_solution(sol_path))

    plt.rcdefaults()
    _fig, ax = plt.subplots(figsize=figsize)

    jobs = sorted(inst.tasks, key=lambda x: inst.nodes[x]["r"])
    start_dates = [_start_date(sol, j) for j in jobs]

    durations = [inst.nodes[j]["p"] for j in jobs]
    y_pos = np.arange(len(jobs))

    ax.barh(y_pos, durations, left=start_dates, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted([inst.nodes[i]["r"] for i in jobs]))
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel("Task risk")
    ax.set_xlabel("Time unit")
    ax.grid(True)
    plt.show()


def linear_regression(x, y):
    """Calculates the best fitting line representing a dataset.
    See: https://realpython.com/linear-regression-in-python/
    """
    return LinearRegression().fit(x, y)


def plot_points_with_best_fit_line(ax, x, y, psize=80, labels=("x", "y")):
    """Plots a scatter graph with a best fit line."""
    ax.grid(True)
    ax.set_xlabel(labels[0]), ax.set_ylabel(labels[1])
    ax.scatter(x, y, s=psize)
    rx = x.reshape((-1, 1))
    ax.plot(x, linear_regression(rx, y).predict(rx))
    print(x, y)
    print("intercept:", linear_regression(rx, y).intercept_)
    print("slope:", linear_regression(rx, y).coef_)
    return ax


def plot_task_policies(instance, sol, figsize=(18, 6)):
    """ "Plots the following graphs together:

    Task risk vs. completion time
    Task duration vs. completion time
    Task distance from depot vs. completion time
    """
    tasks = instance.tasks
    sp = dict(nx.shortest_path_length(instance))
    ys = [
        np.array(x)
        for x in (
            [instance.nodes[i]["r"] for i in tasks],
            [instance.nodes[i]["p"] for i in tasks],
            [sp[0][i] for i in tasks],
        )
    ]
    y_labels = ("Task risk", "Task duration", "Distance between task and depot")

    x = np.array([_completion_date(sol, i) for i in tasks])
    x_label = "Completion time"

    _fig, axs = plt.subplots(ncols=len(ys), figsize=figsize)
    for ax, y, y_label in zip(axs, ys, y_labels, strict=True):
        plot_points_with_best_fit_line(ax, x, y, labels=(x_label, y_label))
