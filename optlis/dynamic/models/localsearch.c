#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define NEUTRALIZING_SPEED 0.3
#define CLEANING_SPEED 0.075

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

struct task {
  int32_t type;
  int32_t site;
  int32_t target;
};

struct instance {
  size_t nnodes;
  size_t ntasks;
  size_t nproducts;
  size_t ntime_units;
  int32_t *nresources;
  int32_t *tasks;
  int32_t *cleaning_start_times;
  int32_t *neutralizing_start_times;
  double *products_risk;
  double *degradation_rates;
  double *metabolizing_rates;
};

struct solution {
  size_t ntasks;
  struct task *task_list;
  double *nodes_concentration;
  double objective;
  int32_t found_at;
};

struct budget {
  int32_t max;
  int32_t consumed;
};

void swap(size_t i, size_t j, struct task *task_list);
double overall_risk(size_t nnodes, size_t nproducts, size_t ntime_units,
                    const double *products_risk,
                    const double *nodes_concentration);
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
  while (try_improve_solution(inst, sol, budget));
  // print_info(inst, sol, budget);
}

void print_info(const struct instance *inst, const struct solution *sol,
                const struct budget *budget) {
  printf("\nInstance:\n");
  printf("%ld nodes, %ld tasks, (%d, %d) resources\n", inst->nnodes,
         inst->ntasks, inst->nresources[0], inst->nresources[1]);
  printf("%ld products, %ld time units\n", inst->nproducts, inst->ntime_units);

  printf("\nTask list:\n");
  for (size_t i = 0; i < sol->ntasks; ++i)
    printf("(%d %d %d) ", sol->task_list[i].type, sol->task_list[i].site,
           sol->task_list[i].target);
  printf("\n");

  printf("\nDegradation rates:\n");
  for (size_t p = 0; p < inst->nproducts; ++p)
    printf("%lf ", inst->degradation_rates[p]);
  printf("\n");

  printf("\nMetabolizing rates:\n");
  for (size_t p = 0; p < inst->nproducts; ++p) {
    for (size_t q = 0; q < inst->nproducts; ++q)
      printf("%lf ", inst->metabolizing_rates[p * inst->nproducts + q]);
    printf("\n");
  }

  printf("\nSolution:\n");
  printf("%lf objective\n", sol->objective);

  printf("\nBudget:\n");
  printf("%d consumed, %d max\n", budget->consumed, budget->max);
}

void swap(size_t i, size_t j, struct task *task_list) {
  struct task taux = task_list[i];
  task_list[i] = task_list[j];
  task_list[j] = taux;
}

double overall_risk(size_t nnodes, size_t nproducts, size_t ntime_units,
                    const double *products_risk,
                    const double *nodes_concentration) {
  double value = 0;
  bool log = true;

  for (size_t i = 1; i < nnodes; ++i) {

    double node_risk = 0;

    for (size_t p = 0; p < nproducts; ++p) {
      // log = nodes_concentration[ntime_units * (p + nproducts * i)] > 0 ? 1 :
      // 0;
      // log = 0;
      // if (log)
      //   printf("[p = %ld] ", p);
      for (size_t t = 0; t < ntime_units; ++t) {
        // https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array
        value += products_risk[p] *
                 nodes_concentration[t + ntime_units * (p + nproducts * i)];
        // if (log)
        //   printf("[%ld]%.2lf ", t,
        //          nodes_concentration[t + ntime_units * (p + nproducts * i)]);
      }
      value += node_risk;
      // if (log)
      //   printf("\n");
    }
    // if (log)
    //   printf("\n");
  }

  return value;
}

void degradation_scheme(const struct instance *inst, struct solution *sol,
                        size_t t) {

  if (t == 0)
    return;

  int nproducts = inst->nproducts;
  int ntime_units = inst->ntime_units;

  for (size_t i = 0; i < inst->nnodes; ++i) {
    for (size_t p = 0; p < inst->nproducts; ++p) {

      double lastc =
          sol->nodes_concentration[(t - 1) + ntime_units * (p + nproducts * i)];
      double currentc = lastc - lastc * inst->degradation_rates[p];

      sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] =
          currentc;
    }
  }
}

void metabolizing_scheme(const struct instance *inst, struct solution *sol,
                         size_t t) {

  if (t == 0)
    return;

  int nproducts = inst->nproducts;
  int ntime_units = inst->ntime_units;

  for (size_t i = 0; i < inst->nnodes; ++i) {
    for (size_t p = 0; p < inst->nproducts; ++p) {
      for (size_t q = 0; q < inst->nproducts; ++q) {
        double lastc =
            sol->nodes_concentration[(t - 1) +
                                     ntime_units * (p + nproducts * i)];
        double amount = (lastc - lastc * inst->degradation_rates[p]) *
                        inst->metabolizing_rates[p * nproducts + q];

        sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] -=
            amount;
        sol->nodes_concentration[t + ntime_units * (q + nproducts * i)] +=
            amount;
      }
    }
  }
}

void neutralizing_scheme(const struct instance *inst, struct solution *sol,
                         size_t i, size_t p, size_t t) {

  if (t == 0)
    return;

  int nproducts = inst->nproducts;
  int ntime_units = inst->ntime_units;

  double lastc =
      sol->nodes_concentration[(t - 1) + ntime_units * (p + nproducts * i)];
  double amount =
      NEUTRALIZING_SPEED * lastc - lastc * inst->degradation_rates[p];

  sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] -= amount;
  sol->nodes_concentration[t + ntime_units * (0 + nproducts * i)] += amount;
}

void cleaning_scheme(const struct instance *inst, struct solution *sol,
                     size_t i, size_t t) {

  if (t == 0)
    return;

  int nproducts = inst->nproducts;
  int ntime_units = inst->ntime_units;

  for (size_t p = 0; p < nproducts; ++p) {
    double currentc =
        MAX(0, sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] -
                   CLEANING_SPEED);
    sol->nodes_concentration[t + ntime_units * (p + nproducts * i)] = currentc;
  }
}

double calculate_schedule(const struct instance *inst, struct solution *sol) {

  // NOTE: assumes the only depot is node 0 and the max number of resources
  // is 64
  bool active_sites[64] = {false}; // used for the no overlap constraints
  int neutralizing_start_times[64] = {0};
  int cleaning_start_times[64] = {0};
  int start_time;
  struct task *neutralizing_resources[64] = {NULL};
  struct task *cleaning_resources[64] = {NULL};

  size_t ntime_units = inst->ntime_units;
  struct task *task = NULL;
  struct task *curr_task = NULL;

  size_t nstarted_tasks = 0;

  for (size_t t = 1; t < ntime_units; ++t) {

    degradation_scheme(inst, sol, t);
    metabolizing_scheme(inst, sol, t);

    // Updates tasks that are currently being processed (neut.)
    for (size_t k = 0; k < inst->nresources[0]; ++k) {

      // Gets the current task of this resource
      curr_task = neutralizing_resources[k];

      if (curr_task != NULL) {

        // Gets the start time of the current task, if it just completed
        start_time =
            inst->neutralizing_start_times[t +
                                           ntime_units * (curr_task->target +
                                                          inst->nproducts *
                                                              curr_task->site)];

        // If resource's current task is not yet complete
        if (neutralizing_start_times[k] >= start_time) {
          // printf("Resource %ldn is working on (%d, %d, %d) at %ld\n", k,
          // curr_task->type, curr_task->site, curr_task->target, t);
          neutralizing_scheme(inst, sol, curr_task->site, curr_task->target, t);
          continue;
        }

        active_sites[curr_task->site] = false;
      }

      // The next task to be schedule
      if (nstarted_tasks < sol->ntasks)
        task = &sol->task_list[nstarted_tasks];

      // This resource is free and can start the next task
      if (task != NULL && task->type == 0 && !active_sites[task->site]) {
        // printf("x_%d_%d_%ld = 1\n", task->site, task->target, t);

        active_sites[task->site] = true;
        neutralizing_resources[k] = task;
        neutralizing_start_times[k] = t;

        neutralizing_scheme(inst, sol, task->site, task->target, t);

        task = NULL; // this task is no longer available to other resources
        ++nstarted_tasks;
      }
    }

    // Updates tasks that are currently being processed (clean.)
    for (size_t k = 0; k < inst->nresources[1]; ++k) {

      // Gets the current task of this resource
      curr_task = cleaning_resources[k];

      if (curr_task != NULL) {

        // Gets the start time of the current task, if it just completed
        start_time =
            inst->cleaning_start_times[curr_task->site * ntime_units + t];

        // If resource's current task is not yet complete
        if (cleaning_start_times[k] >= start_time) {
          // printf("Resource %ldc is working on (%d, %d, %d) at %ld\n", k,
          // curr_task->type, curr_task->site, curr_task->target, t);
          cleaning_scheme(inst, sol, curr_task->site, t);
          continue;
        }

        active_sites[curr_task->site] = false;
      }

      // The next task to be schedule
      if (nstarted_tasks < sol->ntasks)
        task = &sol->task_list[nstarted_tasks];

      // This resource is free and can start the next task
      if (task != NULL && task->type == 1 && !active_sites[task->site]) {

        // printf("y_%d_%ld = 1\n", task->site, t);

        active_sites[task->site] = true;
        cleaning_resources[k] = task;
        cleaning_start_times[k] = t;

        cleaning_scheme(inst, sol, task->site, t);

        task = NULL; // this task is no longer available to other resources
        ++nstarted_tasks;
      }
    }
  }

  sol->objective =
      overall_risk(inst->nnodes, inst->nproducts, inst->ntime_units,
                   inst->products_risk, sol->nodes_concentration);

  return sol->objective;
}
