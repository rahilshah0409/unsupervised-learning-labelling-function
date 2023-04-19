import pickle
import time
import gym
import numpy as np
import waterWorldExperiment.tools as tl
from waterWorldExperiment.trainKMeansClustering import run_clustering, get_random_succ_traces

def user_playing_with_env(env, kmeans_obj, see_replay=True):
    actions, states, times = env.play()

    cluster_labels = kmeans_obj.predict(states)
  
    if see_replay:
        cluster_label_ix = 0
        time_to_sleep = 0.1
        state = env.reset()
        env.render()
        time.sleep(time_to_sleep)
        prev_label = 0
        for j in range(len(actions)):
            action = actions[j]
            t_delta = times[j]
            state, _, _, _ = env.step(action, t_delta)
            label = cluster_labels[cluster_label_ix]
            if prev_label != label:
                print("HAVE WE OVERLAPPED WITH ONE OF THE BALLS?")
            print("Label: {}".format(label))
            env.render()
            prev_label = label
            time.sleep(time_to_sleep)
            cluster_label_ix += 1
            

if __name__ == "__main__":
    kmeans_obj_qbn = pickle.load(open("./trainedClusterObjs/kmeans_qbn.pkl", "rb"))
    kmeans_obj_no_qbn = pickle.load(open("./trainedClusterObjs/kmeans_no_qbn.pkl", "rb"))

    simple_fixed_start_env = gym.make(
        "gym_subgoal_automata:WaterWorldRed-v0",
        params={"generation": "random", "random_restart": False, "environment_seed": 0}
    )
    
    normal_env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "environment_seed": 0},
    )

    fixed_start_env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "random_restart": False, "environment_seed": 0},
    )

    env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": False, "environment_seed": 0},
    )

    fixed_start_env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": False, "environment_seed": 0, "random_restart": False},
    )
    
    # user_playing_with_env(fixed_start_env, kmeans_obj_qbn, see_replay=True)
    ep_durs, state_seqs, events, actions = get_random_succ_traces(normal_env, 1, 1)
    conc_state_seqs = np.concatenate(state_seqs)
    cluster_labels = kmeans_obj_qbn.predict(conc_state_seqs)
    conc_events = np.concatenate(events)
    tl.plot_events_pred_events_from_env_dist(cluster_labels, conc_events, ep_durs)