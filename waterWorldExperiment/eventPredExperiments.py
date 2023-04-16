import time
import numpy as np
from eventPredBaseProcess import runEventPrediction, extract_events
import gym


def vary_no_of_succ_traces(env, no_of_runs, no_of_succ_traces_arr, no_of_eps):
    accuracy_results = [[] for _ in range(len(no_of_runs))]
    for run in no_of_runs:
        for n in no_of_succ_traces_arr:
            accuracy = runEventPrediction(env, n, no_of_eps)
            accuracy_results[run].append(accuracy)
    return accuracy_results


def vary_no_of_episodes(env, no_of_runs, no_of_eps_arr, no_of_succ_traces):
    accuracy_results = [[] for _ in range(len(no_of_runs))]
    for run in no_of_runs:
        for n in no_of_eps_arr:
            accuracy = runEventPrediction(env, no_of_succ_traces, n)
            accuracy_results[run].append(accuracy)
    return accuracy_results

if __name__ == "__main__":
    env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "random_restart": False, "environment_seed": 0},
    )
    no_of_runs = 2
    actions_list = []
    state_seqs = []
    times_per_run = []
    for _ in range(no_of_runs):
        actions, states, times = env.play()
        actions_list.append(actions)
        state_seqs.append(states)
        times_per_run.append(times)

    cluster_labels = extract_events(state_seqs=state_seqs, pairwise_comp=False)
  
    cluster_label_ix = 0
    time_to_sleep = 0.1
    for i in range(no_of_runs):
        state = env.reset()
        env.render()
        time.sleep(time_to_sleep)
        prev_label = 0
        for j in range(len(actions_list[i])):
            action = actions_list[i][j]
            t_delta = times_per_run[i][j]
            state, _, _, _ = env.step(action, t_delta)
            label = cluster_labels[cluster_label_ix]
            if prev_label != label:
                print("HAVE WE OVERLAPPED WITH ONE OF THE BALLS?")
            print("Label: {}".format(label))
            env.render()
            prev_label = label
            time.sleep(time_to_sleep)
            cluster_label_ix += 1

    # no_of_runs = 10
    # env = gym.make(
    #     "gym_subgoal_automata:WaterWorldRedGreen-v0",
    #     params={"generation": "random", "environment_seed": 0},
    # )

    # num_episodes = 500
    # num_succ_traces_arr = np.arange(25, 500, 25)
    # accuracy_results_1 = vary_no_of_succ_traces(
    #     env, no_of_runs, num_succ_traces_arr, num_episodes)

    # num_episodes_arr = np.arange(100, 600, 100)
    # num_succ_traces = 50
    # accuracy_results_1 = vary_no_of_episodes(
    #     env, no_of_runs, num_episodes_arr, num_succ_traces)
