import pulp as plp
import networkx as nx
import click

from instances import load_instance, export_solution


def make_prob(G, model=1):
    """Implements the linear model for the problem."""
    V = G.nodes
    # The set of origins
    O = G.origins
    # The set of 'jobs' to process
    D = G.destinations
    # The number of wt at each origin
    q = nx.get_node_attributes(G, "q")
    # The duration of each job
    p = nx.get_node_attributes(G, "p")
    # The risk at each destination
    r = nx.get_node_attributes(G, "r")
    # The distance between every pair of nodes
    c = dict(nx.shortest_path_length(G))
    # The estimated amount of time periods to process all jobs (T is an upper bound)
    # indexed from 1 to T
    T = G.time_periods

    # Creates the model's variables
    makespan = plp.LpVariable("makespan", lowBound=0, cat=plp.LpInteger)
    sd = plp.LpVariable.dicts("sd", indexs=D, lowBound=0, cat=plp.LpInteger)
    cd = plp.LpVariable.dicts("cd", indexs=D, lowBound=0, cat=plp.LpInteger)
    y = plp.LpVariable.dicts("y", indexs=(V, V, T), cat=plp.LpBinary)

    # The objective function
    prob = plp.LpProblem("Cmax_Pulse_Flow", plp.LpMinimize)

    if model == 1:
        prob += makespan, "Makespan"
    else:
        prob += plp.lpSum(r[i] * cd[i] for i in D)

    # Flow depart from origins
    for i in O:
        prob += (plp.lpSum(y[i][j][t] for t in T
                                      for j in D) <= q[i]
        ), f"R1_Flow_depart_from_origin_{i}"

    # Every destination has to be cleaned at some point
    for j in D:
        prob += (plp.lpSum(y[i][j][t] for t in T
                                      for i in V if i != j) == 1
        ), f"R2_Enter_{j}"

    # Flow conservation constraints
    for j in D:
        for t in T:
            prob += (
                plp.lpSum(y[i][j][t - c[i][j] - p[j]] for i in V if i != j
                                                      if t - c[i][j] - p[j] >= T[0])
                == plp.lpSum(y[j][i][t] for i in V if i != j)
            ), f"R4_Flow_conservation_on_{j}_at_period_{t}"

    # Calculates the start time of every node
    for j in D:
        prob += (sd[j] == plp.lpSum(t * y[i][j][t] for t in T
                                                   for i in V if i != j)
        ), f"R5_Start_date_of_{j}"

    # Precedence constraints
    if model == 1:
        for i, j in G.precedencies:
            prob += sd[i] <= sd[j], f"R6_Start_cleaning_{i}_before_{j}"

    # Calculates the completion time of every node
    for j in D:
        prob += (cd[j] == plp.lpSum((t + c[i][j]) * y[i][j][t] for t in T
                                                               for i in V if i != j)
                          + p[j]
        ), f"R7_Completion_date_of_{j}"

    # Calculates the makespan
    for j in D:
        prob += makespan >= cd[j], f"R8_Cmax_geq_C_{j}"

    return prob


@click.command()
@click.argument("instance-path")
@click.option("--model", type=int, default=1,
              help="Chose the model to run, choices are 1 (default) or 2.")
@click.option("--time-limit",
              help="The maximum time limit for the execution (in seconds).")
@click.option("--log-path", help="File to write the execution log.")
@click.option("--sol-path", help="File to write the solution (in json).")
def run(instance_path, model=1, time_limit=None, log_path=None, sol_path=None):
    """Runs the model from command line.

       USAGE:

       python cmax_pulse_flow.py [OPTIONS] INSTANCE_PATH

       OPTIONS:

       --help\n
            show this message and exit
       --model[=MODEL]\n
            the model to run. Choices are 1 (default) or 2\n
       --time-limit[=LIMIT]\n
            the maximum time limit for the execution (in seconds)\n
       --log-path[=LOG_PATH]\n
            where to write the execution log\n
       --sol-path[=SOL_PATH]\n
            where to write the solution (in Pulp json format)\n
    """
    if model not in (1, 2):
        raise ValueError("Chosen model is invalid. Please choose 1 or 2")

    G = load_instance(instance_path)
    prob = make_prob(G, model=model)
    prob.writeLP("CmaxPulseFlow.lp")

    # Solves the problem with CPLEX (assuming CPLEX is availible)
    solver = plp.getSolver('CPLEX_CMD', timeLimit=time_limit, logPath=log_path)
    prob.solve(solver)
    prob.roundSolution()

    # Prints variables with it's resolved optimum value
    print("")
    for v in prob.variables():
        if v.varValue:
            print(v.name, "=", v.varValue)

    if sol_path:
        export_solution(sol_path, instance_path,
                        {v.name: v.varValue for v in prob.variables()})


if __name__ == "__main__":
    run()
