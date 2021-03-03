from pathlib import Path
from dataclasses import dataclass


@dataclass
class Node:
    id: int
    type: int
    p: int
    q: int
    r: float


@dataclass
class Arc:
    i: Node
    j: Node


@dataclass
class Instance:
    V: list[Node]
    A: list[Arc]
    O: list[Node]
    D: list[Node]
    T: int


def loads(path):
    data = {
        "V": [],
        "A": [],
        "O": [],
        "D": [],
        "T": 0,
    }

    with open(path, "r") as f:
        nb_nodes = int(f.readline())
        for _ in range(nb_nodes):
            id, type, p, q, r = f.readline().split()
            node = Node(int(id), int(type), int(p), int(q), float(r))
            data["V"].append(node)
        nb_arcs = int(f.readline())
        for _ in range(nb_arcs):
            i, j = f.readline().split()
            data["A"].append(Arc(int(i), int(j)))
            data["A"].append(Arc(int(j), int(i)))
        data["T"] = sum(v.p for v in data["V"])
        data["O"] = [v for v in data["V"] if v.type == 0]
        data["D"] = [v for v in data["V"] if v.type == 1]
    return data
