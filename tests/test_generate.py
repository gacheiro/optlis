from optlis import generate_instance


def test_generate_instance():
    """Tests the function to generate a problem instance."""
    # Generates an homogeneous risk hexagonal instance
    inst = generate_instance(size=(3, 2), nb_teams=1, seed=0)
    assert inst.depots == [0]
    assert inst.tasks == list(range(1, 8))
    for i in inst.tasks:
        inst.nodes[i]["r"] = 0.5
    assert list(inst.time_periods)[-1] == 30
