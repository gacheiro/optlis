import numpy as np

from optlis import generate_instance


def test_generate_instance():
    """Tests the function to generate a problem instance."""
    inst = generate_instance(size=(3, 2), nb_teams=1, seed=0)
    assert np.array_equal(inst.depots, np.array([0]))
    assert np.array_equal(inst.tasks, np.array([1, 2, 3, 4, 5, 6, 7]))
    assert list(inst.time_periods)[-1] == 55
