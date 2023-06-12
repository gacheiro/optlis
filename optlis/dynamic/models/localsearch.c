#include<stdint.h>
#include<limits.h>
#include<stdbool.h>
#include<stdio.h>

#define CLEANING_SPEED 0.15

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

struct instance {
  size_t nnodes;
  size_t ntasks;
  size_t nproducts;
  size_t nresources;
  size_t ntime_units;
  int32_t *tasks;
  int32_t *nodes_duration;
  double *products_risk;
  double *degration_rates;
};

struct solution {
  size_t ntasks;
  int32_t *task_list;
  double *nodes_concentration;
  double objective;
  int32_t found_at;
};

struct budget {
  int32_t max;
  int32_t consumed;
};

void swap(size_t i, size_t j, int32_t *task_list);
double overall_risk(size_t nnodes, size_t nproducts, size_t ntime_units,
                    const double *products_risk, const double *nodes_concentration);
double calculate_schedule(const struct instance *inst, struct solution *sol);
bool try_improve_solution(const struct instance *inst, struct solution *sol,
                          struct budget *budget);
void local_search(const struct instance *inst, struct solution *sol,
                  struct budget *budget);
void print_info(const struct instance *inst, const struct solution *sol,
                const struct budget *budget);

bool try_improve_solution(const struct instance *inst, struct solution *sol,
                          struct budget *budget) {
  double current_objective = sol->objective;

  for (size_t i = 0; i < sol->ntasks; ++i) {
    for (size_t j = i + 1; j < sol->ntasks; ++j) {

      if (budget->consumed >= budget->max)
        return false;

      swap(i, j, sol->task_list);

      calculate_schedule(inst, sol);
      budget->consumed++;

      if (sol->objective < current_objective) {
        sol->found_at = budget->consumed;
        return true;
      }

      swap(i, j, sol->task_list);
      sol->objective = current_objective;
    }
  }

  return false;
}

void local_search(const struct instance *inst, struct solution *sol,
                  struct budget *budget) {
  calculate_schedule(inst, sol);
  print_info(inst, sol, budget);
}

void print_info(const struct instance *inst, const struct solution *sol,
                const struct budget *budget) {
  printf("Instance:\n");
  printf("%ld nodes, %ld tasks, %ld resources\n", inst->nnodes,
         inst->ntasks, inst->nresources);
  printf("%ld products, %ld time units\n", inst->nproducts,
         inst->ntime_units);

  printf("\nDegradation rates:\n");
  for (size_t p = 0; p < inst->nproducts; ++p)
    printf("%lf ", inst->degration_rates[p]);
  printf("\n");

  printf("\nSolution:\n");
  printf("%lf objective\n", sol->objective);

  printf("\nBudget:\n");
  printf("%d consumed, %d max\n", budget->consumed, budget->max);
}

void swap(size_t i, size_t j, int32_t *task_list) {
  int32_t taux = task_list[i];
  task_list[i] = task_list[j];
  task_list[j] = taux;
}

double overall_risk(size_t nnodes, size_t nproducts, size_t ntime_units,
                    const double *products_risk, const double *nodes_concentration) {
  double value = 0;
  bool log;
  for (size_t i = 1; i < nnodes; ++i) {

    double node_risk = 0;
  
    for (size_t p = 0; p < nproducts; ++p) {
      log = nodes_concentration[ntime_units * (p + nproducts * i)] > 0 ? 1 : 0;
      if (log) printf("[p = %ld] ", p);
      for (size_t t = 0; t < ntime_units; ++t) {
        // https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array
        value += products_risk[p] * nodes_concentration[t + ntime_units * (p + nproducts * i)];
        if (log) printf("[%ld]%.2lf ", t, nodes_concentration[t + ntime_units * (p + nproducts * i)]);
      }
      value += node_risk;
      if (log) printf("\n");
    }
    if (log) printf("\n");
  }

  return value;
}

void product_degradation(const struct instance *inst, struct solution *sol, size_t t) {

  if (t == 0)
    return;

  int nproducts = inst->nproducts;
  int ntime_units = inst->ntime_units;

  for (size_t i = 0; i < inst->nnodes; ++i) {
    for (size_t p = 0; p < inst->nproducts; ++p) {

      double lastc = sol->nodes_concentration[(t - 1) + ntime_units * (p + nproducts * i)];
      double currentc = lastc - lastc*inst->degration_rates[p];

      sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] = currentc;
    }
  }
}

void product_cleaning(const struct instance *inst, struct solution *sol, size_t i, size_t t) {
  
  if (t == 0)
    return;

  int nproducts = inst->nproducts;
  int ntime_units = inst->ntime_units;

  for (size_t p = 0; p < nproducts; ++p) {

    double currentc = MAX(0, sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] - CLEANING_SPEED);
    sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] = currentc;
  }
}

double calculate_schedule(const struct instance *inst, struct solution *sol) {

  // NOTE: assumes the only depot is node 0 and the max number of resources is 64
  
  int start_times[64] = {0};
  int finish_times[64] = {0};
  int resources[64] = {0};
  
  size_t ntime_units = inst->ntime_units;

  int ntasks = inst->ntasks;
  size_t task_id = -1;
  size_t processed_tasks = 0;

  for (size_t time_unit = 1; time_unit < ntime_units; ++time_unit) {

    product_degradation(inst, sol, time_unit);

    for (size_t k = 0; k < inst->nresources; ++k) {

      // Gets the current task of this resource
      size_t curr_task_id = resources[k];
      // Gets the start time of the current task, if it just completed
      int start_time = inst->nodes_duration[curr_task_id*ntime_units + time_unit];

      // If resource's current task is not yet complete
      if (start_times[curr_task_id] > start_time) {
        printf("Cleaning task %ld at %ld\n", curr_task_id, time_unit);
        product_cleaning(inst, sol, curr_task_id, time_unit);
        continue;
      }

      // Else, allocates this resource to process task_id (if exists)  
      if (processed_tasks < ntasks)
        task_id = sol->task_list[processed_tasks];
      else
        task_id = -1;

      if (task_id != -1) {
        printf("Allocating resource %ld to task %ld at %ld\n", k, task_id, time_unit);
        resources[k] = task_id;
        start_times[task_id] = time_unit;
        product_cleaning(inst, sol, task_id, time_unit);

        // Advances to the next task
        ++processed_tasks;
      }
    }
  }

  sol->objective = overall_risk(
    inst->nnodes,
    inst->nproducts,
    inst->ntime_units,
    inst->products_risk,
    sol->nodes_concentration
  );

  return sol->objective;
}
