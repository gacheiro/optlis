import networkx as nx


class Graph(nx.Graph):
    """Subclass of nx.Graph class with some utility properties."""
    
    @property
    def origins(self):
        # Note: is there a better way to do this?
        return [n for n in self.nodes if self.nodes[n]["type"] == 0]

    @property    
    def destinations(self):
        return [n for n in self.nodes if self.nodes[n]["type"] == 1]

    @property
    def time_periods(self):
        sumT = sum(self.nodes[n]["p"] for n in self.nodes())
        return list(range(1, sumT+1))


def loads(path):
    """Loads an instance from a file."""
    nodes = []
    edges = []
    with open(path, "r") as f:
        nb_nodes = int(f.readline())
        for _ in range(nb_nodes):
            id, type, p, q, r = f.readline().split()
            nodes.append((int(id), {
                "type": int(type),
                "p": int(p),
                "q": int(q),
                "r": float(r),
            }))

        nb_edges = int(f.readline())
        for _ in range(nb_edges):
            i, j = [int(u) for u in f.readline().split()]
            edges.append((i, j))
    
    G = Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def save(G, path):
    """Saves an instance to a file."""
    nb_nodes = len(G.nodes)
    nb_edges = len(G.edges)
    with open(path, "w") as f:
        f.write(f"{nb_nodes}\n")        
        for (id, data) in G.nodes(data=True):
            type, p, q, r = (data["type"],
                             data["p"],
                             data["q"],
                             data["r"])
            f.write(f"{id} {type} {p} {q} {r:.1f}\n")
        f.write(f"{nb_edges}\n")
        for (i, j) in G.edges():
            f.write(f"{i} {j}\n")
