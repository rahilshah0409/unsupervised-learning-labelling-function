import pickle
import time
import gym
import numpy as np
import sys
sys.path.append("../../..")
import waterWorld.utils.tools as tl
from waterWorld.clustering.clusteringTraining import extract_shortest_succ_traces, get_random_succ_trace, get_random_succ_traces, train_clustering, encode_state_seq

def user_playing_with_env(env, kmeans_obj, see_replay=True):
    actions, states, _, times = env.play()

    cluster_labels = kmeans_obj.predict(states)
    print(states[-4])
  
    if see_replay:
        cluster_label_ix = 0
        time_to_sleep = 0.5
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
        ep_dur = len(states)

    return ep_dur, states, events


def evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, event_labels, ep_dur):
    precision_scores, recall_scores = tl.compare_changes_in_cluster_ids_vs_events(cluster_labels_arr, event_labels, ep_dur)
    print(subplot_titles)
    print("Precision scores:")
    print(precision_scores)
    print("Recall scores:")
    print(recall_scores)

    if len(cluster_labels_arr) == 1:
        tl.visualise_cluster_labels_vs_events(cluster_labels_arr[0], subplot_titles, event_labels, ep_dur)
    else:
        tl.visualise_cluster_labels_arr_vs_events(cluster_labels_arr, subplot_titles, event_labels, ep_dur)



# Try out different forms of clustering. Not implemented yet

# Varying number of clusters that we aim to extract and how we extract event labels given this. Not implemented yet
def vary_no_of_clusters(env, num_succ_traces, num_eps, num_clusters_arr):
    ep_dur, states, events = get_test_trace(env, random_gen=True)
    cluster_labels_arr = []
    _, state_seqs, _, _ = get_random_succ_traces(env, num_succ_traces, num_eps)

    for num_clusters in num_clusters_arr:
        kmeans_obj, _ = train_clustering(state_seqs, num_clusters, encode_states=False)
        cluster_labels = kmeans_obj.predict(states)
        cluster_labels_arr.append(cluster_labels)

    subplot_titles = [str(n) for n in num_clusters_arr]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, events, ep_dur)

# See if encoding the states with an autoencoder improves association between cluster labels and event labels
def affect_of_autoencoder(env, num_succ_traces, num_eps, use_velocities, activations):
    ep_dur, states, events = get_test_trace(env, random_gen=False)
    print("Episode duration is: {}".format(ep_dur))
    cluster_labels_arr = []
    _, state_seqs, _, _ = get_random_succ_traces(env, num_succ_traces, num_eps)

    kmeans_obj_no_qbn, _ = train_clustering(state_seqs, 4, use_velocities, None, encode_states=False)
    cluster_labels_no_qbn = kmeans_obj_no_qbn.predict(states)
    print(len(cluster_labels_no_qbn))
    cluster_labels_arr.append(cluster_labels_no_qbn)

    for i in range(len(activations)):
        activation = activations[i]
        kmeans_obj_qbn, qbn = train_clustering(state_seqs, 4, use_velocities, activation, encode_states=True)
        encoded_states = encode_state_seq(qbn, states)
        cluster_labels_qbn = kmeans_obj_qbn.predict(encoded_states)
        print(len(cluster_labels_qbn))
        cluster_labels_arr.append(cluster_labels_qbn)

    subplot_titles = ["Without QBN", "Binary sigmoid", "Tanh", "Sigmoid"]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, events, ep_dur)

# Affect of different number of successful traces
def vary_no_of_succ_traces(env, num_succ_traces_arr, use_velocities, ep_durs, states, events, actions):
    ep_dur, test_trace, test_event_labels = get_test_trace(env, random_gen=True)
    cluster_labels_arr = []
    for num_succ_traces in num_succ_traces_arr:
        _, state_seqs, _, _ = extract_shortest_succ_traces(ep_durs, states, events, actions, num_succ_traces)
        kmeans_obj, _ = train_clustering(state_seqs, 4, use_velocities, encode_states=False)
        cluster_labels = kmeans_obj.predict(test_trace)
        cluster_labels_arr.append(cluster_labels)
    
    subplot_titles = [str(n) for n in num_succ_traces_arr]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, test_event_labels, ep_dur)

# Affect of different number of total episodes
# Do we want to fix the random traces that are created before the shortest ones are considered?
def vary_no_of_eps(env, num_eps_arr, num_succ_traces, use_velocities):
    ep_dur, states, events = get_test_trace(env, random_gen=True)
    cluster_labels_arr = []
    for num_eps in num_eps_arr:
        _, state_seqs, _, _ = get_random_succ_traces(env, num_succ_traces, num_eps)
        kmeans_obj, _ = train_clustering(state_seqs, 4, use_velocities, "binarySigmoid", encode_states=False)
        cluster_labels = kmeans_obj.predict(states)
        cluster_labels_arr.append(cluster_labels)
    
    subplot_titles = [str(n) for n in num_eps_arr]
    evaluate_cluster_label_prediction(cluster_labels_arr, subplot_titles, events, ep_dur)