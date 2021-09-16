from pathlib import Path

import pytest
import networkx as nx

from instances import load_instance, grid


@pytest.mark.skip(reason="no way of currently testing this.")
def test_grid2d(instance_grid3x3_data):
    """Tests the function to generate grid instances.
       It tries to generate the same `grid3x3.dat` instance.
    """
    G = grid()
    assert list(G.nodes(data=True)) == instance_grid3x3_data[0]
    assert set(G.edges) == set(instance_grid3x3_data[1])
