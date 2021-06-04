from pathlib import Path

from instances import load_instance, import_solution


def test_load_instance(instance_grid3x3_data):
    """Tests the function to load an instance from a file."""
    G = load_instance(Path("data/instances/example.dat"))
    assert list(G.nodes(data=True)) == instance_grid3x3_data[0]
    assert set(G.edges) == set(instance_grid3x3_data[1])


def test_Graph():
    """Tests the Graph class."""
    G = load_instance(Path("data/instances/example.dat"))
    assert list(G.origins) == [0, 1]
    assert list(G.destinations) == [2, 3, 4, 5, 6, 7, 8]
    assert list(G.time_periods)[-1] == 36 # NOTE: see G.time_periods
    assert set(G.precedencies) == {       # docstring for the formula
        (8, 7), (7, 6), (7, 5),
        (6, 4), (5, 4), (4, 3), (3, 2),
    }


def test_import_solution():
    """"Tests the function to import a solution from a file."""
    assert import_solution("data/solutions/example.sol") == dict(
        # Completion date
        cd_2=28,
        cd_3=23,
        cd_4=21,
        cd_5=16,
        cd_6=14,
        cd_7=8,
        cd_8=10,
        makespan=28,
        # Start date
        sd_2=21,
        sd_3=16,
        sd_4=14,
        sd_5=10,
        sd_6=8,
        sd_7=1,
        sd_8=1,
        # Flows
        y_0_8_1=1,
        y_1_7_1=1,
        y_2_0_28=1,
        y_3_0_23=1,
        y_4_2_21=1,
        y_5_3_16=1,
        y_6_4_14=1,
        y_7_6_8=1,
        y_8_5_10=1,
    )
