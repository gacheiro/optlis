import pulp as plp
import networkx as nx
import click

from instances.inst import loads


def make_prob(V=[], O=[], D=[], T=[], p={}, q={}):
    """Implement a model based on Christofides et al (1987) for the RCPSP,
       except the precendence constraints.
    """
    prob = plp.LpProblem("Cmax_Pulse_Flow", plp.LpMinimize)

    # Creates the model's variables
    Cmax = plp.LpVariable("Cmax", lowBound=0, cat=plp.LpInteger)
    x = plp.LpVariable.dicts("x", indexs=(V, V, T), cat=plp.LpBinary)
    C = plp.LpVariable.dicts("C", indexs=D, lowBound=0, cat=plp.LpInteger)

    # Add the objective function to 'prob'
    prob += Cmax, "Makespan"

    # Flow depart from origins
    for i in O:
        prob += (plp.lpSum([x[i][j][t] for t in T
                                       for j in D]) <= q[i],
                f"Flow_depart_from_origin_{i}")
    '''
    # Ensure the maximum number of resources is respected
    for t in T:
        prob += (plp.lpSum([x[i][j][tau] for j in D
                                         for i in V
                                         for tau in range(t - p[i] + 1, t+1)
                                         if tau >= 1])
                <= sum(q.values()),
                f"Resource_constraint_at_period_{t}")
    '''
    # Flow conservation constraints
    for j in D:
        for t in T:
            if t - p[j] <= 0:
                continue
            prob += (plp.lpSum([x[i][j][t-p[j]] for i in V])
                    - plp.lpSum([x[j][i][t] for i in V]) == 0,
                    f"Flow_conservation_on_{j}_at_periods_{t-p[j]}_{t}")

    # Every destination has to be cleaned at some point (1)
    for j in D:
        prob += (plp.lpSum([x[i][j][t] for t in T
                                       for i in V]) == 1,
                f"Clean_destination_arrive_{j}")
    
    # Every destination has to be cleaned at some point (2)
    for j in D:
        prob += (plp.lpSum([x[j][i][t] for t in T
                                       for i in V]) == 1,
                f"Clean_destination_depart_{j}")

    # Calculates the completion time of every node
    for j in D:
        prob += (C[j] - plp.lpSum([t*x[i][j][t] for t in T
                                                for i in V]) == p[j],
                f"Completion_time_of_{j}")

    # Calculates the makespan
    for j in D:
        prob += Cmax - C[j] >= 0, f"Cmax_geq_C_{j}"

    return prob


@click.command()
@click.argument("path")
def solve(path):
    G = loads(path)
    V = G.nodes()
    # The set of origins
    O = list(n for n in G.nodes() if G.nodes[n]["type"] == 0)
    # The set of 'jobs' to process
    D = list(n for n in G.nodes() if G.nodes[n]["type"] == 1)
    # The estimated amount of time neeeded to process all jobs (an upper bound)
    # indexed from 1 to T
    sumT = sum(G.nodes[n]["p"] for n in G.nodes())
    T = list(range(1, sumT+1))
    # The duration of each job
    p = nx.get_node_attributes(G, "p")
    # The number of wt at each origin
    q = nx.get_node_attributes(G, "q")

    # Build and solve the problem using instance data
    prob = make_prob(V, O, D, T, p, q)
    # The problem is written to an .lp file
    prob.writeLP("CmaxPulseFlow.lp")
    prob.solve()
    
    # Print variables with it's resolved optimum value
    for v in prob.variables():
        if v.varValue > 0:
            print(v.name, "=", v.varValue)

    # Print the optimum objective function value   
    print("Makespan = ", plp.value(prob.objective))


if __name__ == "__main__":
    solve()
