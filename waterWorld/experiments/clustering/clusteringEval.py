import pickle
import time
import gym
import numpy as np
import waterWorld.utils.tools as tl
from waterWorld.clustering.clusteringTraining import get_random_succ_trace, get_random_succ_traces, train_clustering

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


def evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, event_labels, ep_dur):
    precision_scores, recall_scores = tl.compare_changes_in_cluster_ids_vs_events(cluster_labels_arr, event_labels, ep_dur)
    print("Precision scores:")
    print(precision_scores)
    print("Recall scores:")
    print(recall_scores)


# Try out different forms of clustering. Not implemented yet

# See if encoding the states with an autoencoder improves association between cluster labels and event labels
def affect_of_autoencoder(env, num_succ_traces, num_eps):
    ep_dur, states, events = get_test_trace(env, random_gen=True)
    cluster_labels_arr = []
    _, state_seqs, _, _ = get_random_succ_traces(env, num_succ_traces, num_eps)

    kmeans_obj_no_qbn = train_clustering(state_seqs, 4, encode_states=False)
    cluster_labels_no_qbn = kmeans_obj_no_qbn.predict(states)
    cluster_labels_arr.append(cluster_labels_no_qbn)

    kmeans_obj_qbn = train_clustering(state_seqs, 4, encode_states=True)
    cluster_labels_qbn = kmeans_obj_qbn.predict(states)
    cluster_labels_arr.append(cluster_labels_qbn)

    subplot_titles = ["Without QBN", "With QBN"]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, events, ep_dur)

# Affect of different number of successful traces
def vary_no_of_succ_traces(env, num_succ_traces_arr, num_eps):
    ep_dur, states, events = get_test_trace(env, random_gen=True)
    cluster_labels_arr = []
    for num_succ_traces in num_succ_traces_arr:
        _, state_seqs, _, _ = get_random_succ_traces(env, num_succ_traces, num_eps)
        kmeans_obj = train_clustering(state_seqs, 4)
        cluster_labels = kmeans_obj.predict(states)
        cluster_labels_arr.append(cluster_labels)
    
    subplot_titles = [str(n) for n in num_succ_traces_arr]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, events, ep_dur)

# Affect of different number of total episodes
def vary_no_of_eps(env, num_eps_arr, num_succ_traces):
    ep_dur, states, events = get_test_trace(env, random_gen=True)
    cluster_labels_arr = []
    for num_eps in num_eps_arr:
        _, state_seqs, _, _ = get_random_succ_traces(env, num_succ_traces, num_eps)
        kmeans_obj = train_clustering(state_seqs, 4)
        cluster_labels = kmeans_obj.predict(states)
        cluster_labels_arr.append(cluster_labels)
    
    subplot_titles = [str(n) for n in num_eps_arr]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, events, ep_dur)