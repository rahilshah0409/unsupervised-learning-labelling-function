import pickle
import time
import gym
import numpy as np
import tools as tl
from trainKMeansClustering import get_random_succ_trace, encode_state_seqs

def user_playing_with_env(env, kmeans_obj, see_replay=True):
    actions, states, _, times = env.play()

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
            

def get_test_trace(env, random_gen=True):
    ep_dur = 0
    states = []
    events = []
    if random_gen:
        ep_dur, states, events, _ = get_random_succ_trace(env)
    else:
        _, states, events, _ = env.play()

    return ep_dur, states, events

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
    ep_dur, state_seqs, events = get_test_trace(normal_env)
    qbn = tl.loadSavedQBN("./trainedQBN/finalModel.pth")
    encoded_state_seqs = encode_state_seqs(qbn, state_seqs)

    conc_state_seqs = np.concatenate(state_seqs)
    cluster_labels_no_qbn = kmeans_obj_no_qbn.predict(conc_state_seqs)
    conc_encoded_state_seqs = np.concatenate(encoded_state_seqs)
    cluster_labels_qbn = kmeans_obj_qbn.predict(conc_encoded_state_seqs)

    conc_events = np.concatenate(events)
    tl.visualise_cluster_labels_vs_events(cluster_labels_no_qbn, cluster_labels_qbn, conc_events, ep_dur)

    p_no_qbn, r_no_qbn, p_qbn, r_qbn = tl.compare_changes_in_cluster_ids_vs_events(cluster_labels_no_qbn, cluster_labels_qbn, conc_events, ep_dur)

    print("Scores without using the QBN. Precision: {}. Recall: {}".format(p_no_qbn, r_no_qbn))
    print("Scores when QBN is used. Precision: {}. Recall: {}".format(p_qbn, r_qbn))