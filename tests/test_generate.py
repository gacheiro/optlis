from optlis import generate_instance


def test_generate_instance():
    """Tests the function to generate a problem instance."""
    # Generates an homogeneous risk hexagonal instance
    G = generate_instance(size=(3, 2), nb_origins=1, q=1, topology="hexagonal",
                          risk_distribution="homogeneous")
    assert G.origins == [0]
    assert G.destinations == list(range(1, 8))
    for i in G.destinations:
        G.nodes[i]["r"] = 0.5
    assert list(G.time_periods)[-1] == 30
