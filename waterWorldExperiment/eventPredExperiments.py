import numpy as np
from eventPredBaseProcess import runEventPrediction


def vary_no_of_succ_traces(no_of_runs, no_of_succ_traces_arr, no_of_eps):
    accuracy_results = [[] for _ in range(len(no_of_runs))]
    for run in no_of_runs:
        for n in no_of_succ_traces_arr:
            accuracy = runEventPrediction(n, no_of_eps)
            accuracy_results[run].append(accuracy)
    return accuracy_results


def vary_no_of_episodes(no_of_runs, no_of_eps_arr, no_of_succ_traces):
    accuracy_results = [[] for _ in range(len(no_of_runs))]
    for run in no_of_runs:
        for n in no_of_eps_arr:
            accuracy = runEventPrediction(no_of_succ_traces, n)
            accuracy_results[run].append(accuracy)
    return accuracy_results


if __name__ == "__main__":
    no_of_runs = 10

    num_episodes = 500
    num_succ_traces_arr = np.arange(25, 500, 25)
    accuracy_results_1 = vary_no_of_succ_traces(
        no_of_runs, num_succ_traces_arr, num_episodes)

    num_episodes_arr = np.arange(100, 600, 100)
    num_succ_traces = 50
    accuracy_results_1 = vary_no_of_episodes(
        no_of_runs, num_episodes_arr, num_succ_traces)
