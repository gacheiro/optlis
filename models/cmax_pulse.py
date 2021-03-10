import pulp as plp
import networkx as nx
import click

from instances import loads


def make_prob(D=[], T=[], p={}, Q=0):
    """Implement the model from Christofides et al (1987) for the RCPSP,
       except the precendence constraints.
    """
    prob = plp.LpProblem("Cmax_Pulse", plp.LpMinimize)

    # Creates the model's variables
    Cmax = plp.LpVariable("Cmax", lowBound=0, cat=plp.LpInteger)
    x = plp.LpVariable.dicts("x", indexs=(D, T), cat=plp.LpBinary)
    S = plp.LpVariable.dicts("S", indexs=D, lowBound=0, cat=plp.LpInteger)
    C = plp.LpVariable.dicts("C", indexs=D, lowBound=0, cat=plp.LpInteger)

    # Add the objective function to 'prob'
    prob += Cmax, "Makespan"

    # Ensure the maximum number of resources is respected
    for t in T:
        prob += (plp.lpSum([x[i][tau] for i in D
                                      for tau in range(t - p[i] + 1, t+1)
                                      if tau >= 1])
                <= Q,
                f"Resource_constraint_at_period_{t}")

    # Every destination has to be cleaned at some point
    for i in D:
        prob += plp.lpSum([x[i][t] for t in T]) == 1, f"Clean_destination_{i}"

    # Calculates the start time of the cleaning at every node
    for i in D:
        prob += S[i] - plp.lpSum([t*x[i][t] for t in T]) == 0, f"Start_time_of_{i}"

    # Calculates the completion time of every node
    for i in D:
        prob += C[i] - S[i] == p[i], f"Completion_time_of_{i}"

    # Calculates the makespan
    for i in D:
        prob += Cmax - C[i] >= 0, f"Cmax_geq_C_{i}"

    return prob


@click.command()
@click.argument("path")
def solve(path):
    G = loads(path)
    # The set of 'jobs' to process
    D = list(n for n in G.nodes() if G.nodes[n]["type"] == 1)
    # The number of machines to process the jobs
    Q = sum(G.nodes[i]["q"] for i in G.nodes())
    # The estimated amount of time neeeded to process all jobs (an upper bound)
    # indexed from 1 to T
    sumT = sum(G.nodes[n]["p"] for n in G.nodes())
    T = list(range(1, sumT+1))
    # The duration of each job
    p = nx.get_node_attributes(G, "p")

    # Build and solve the problem using instance data
    prob = make_prob(D, T, p, Q)
    # The problem is written to an .lp file
    prob.writeLP("CmaxPulse.lp")
    prob.solve()
    
    # Print variables with it's resolved optimum value
    for v in prob.variables():
        if v.varValue > 0:
            print(v.name, "=", v.varValue)

    # Print the optimum objective function value   
    print("Makespan = ", plp.value(prob.objective))


if __name__ == "__main__":
    solve()
