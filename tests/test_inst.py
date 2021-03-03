from pathlib import Path

import pytest

from instances.inst import loads, Arc, Node, Instance


@pytest.fixture(scope="session")
def instance_data():
    return loads(Path("data/instances/example.dat"))


def test_loads(instance_data):
    for key in ("V", "A", "O", "D", "T"):
        assert key in instance_data


def test_instance(instance_data):
    instance = Instance(**instance_data)
    assert instance.V == [
        Node(id=0, type=0, p=0, q=1, r=0.0),
        Node(id=1, type=1, p=2, q=0, r=0.8),
        Node(id=2, type=1, p=1, q=0, r=0.3),
        Node(id=3, type=1, p=1, q=0, r=0.4),
        Node(id=4, type=1, p=1, q=0, r=0.2),
        Node(id=5, type=1, p=1, q=0, r=0.75),
        Node(id=6, type=0, p=0, q=1, r=0.0),
    ]
    assert instance.A == [
        Arc(0, 1), Arc(1, 0),
        Arc(0, 2), Arc(2, 0),
        Arc(1, 2), Arc(2, 1),
        Arc(1, 3), Arc(3, 1),
        Arc(2, 3), Arc(3, 2),
        Arc(2, 5), Arc(5, 2),
        Arc(3, 4), Arc(4, 3),
        Arc(3, 6), Arc(6, 3),
        Arc(4, 5), Arc(5, 4),
        Arc(4, 6), Arc(6, 4),
    ]
    assert instance.O == [
        Node(id=0, type=0, p=0, q=1, r=0.0),
        Node(id=6, type=0, p=0, q=1, r=0.0),
    ]
    assert instance.D == [
        Node(id=1, type=1, p=2, q=0, r=0.8),
        Node(id=2, type=1, p=1, q=0, r=0.3),
        Node(id=3, type=1, p=1, q=0, r=0.4),
        Node(id=4, type=1, p=1, q=0, r=0.2),
        Node(id=5, type=1, p=1, q=0, r=0.75),
    ]
    assert instance.T == 6
