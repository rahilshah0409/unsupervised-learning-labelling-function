import os
import pickle
import torch.nn as nn
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import torch
import gym
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from waterWorld.qbnTraining.quantisationMethods import BinarySigmoid
import waterWorld.utils.tools as tl


# Converts string ID into correct filename where the relevant autoencoder can be loaded
def find_qbn_filename(activation, use_velocities):
    qbn_prefix = ""
    if activation == "binarySigmoid":
        qbn_prefix = "binSig"
    elif activation == "tanh":
        qbn_prefix = "tanh"
    elif activation == "sigmoid":
        qbn_prefix = "sig"
    qbn_suffix = "Normal.pth" if use_velocities else "Static.pth"
    qbn_filename = qbn_prefix + qbn_suffix
    return qbn_filename


def convert_activation(str):
    if str == "binarySigmoid":
        return BinarySigmoid()
    elif str == "tanh":
        return nn.Tanh()
    elif str == "sigmoid":
        return nn.Sigmoid()


# Defines a policy for an agent to choose actions at each timestep. Initially random
def choose_action():
    NUM_ACTIONS = 5
    action_probs = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
    action = random.choices(range(NUM_ACTIONS), weights=action_probs, k=1)[0]
    return action


# Runs an event with a random policy and keeps track of successful traces and the states and episodes observed in
# each trace
def run_agent(env, num_episodes):
    episode_durations = []
    states_traversed = []
    events_per_episode = []
    actions_per_episode = []
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))

        state = env.reset()
        states = []
        states.append(state)
        events_observed = [set()]
        done, terminated, t = False, False, 1
        actions = []

        while not (done or terminated):
            action = choose_action()
            actions.append(action)
            next_state, _, done, observations = env.step(action, t)
            state = next_state
            states.append(state)
            t += 1

            # if (observations != set()):
            events_observed.append(observations)
            if done:
                episode_durations.append((t, ep))

        actions_per_episode.append(actions)
        states_traversed.append(states)
        events_per_episode.append(events_observed)
    return episode_durations, states_traversed, events_per_episode, actions_per_episode


# Extracts the n shortest successful traces, where n is no_of_traces_to_extract
def extract_shortest_succ_traces(
        trace_lengths, states, events, actions, no_of_traces_to_extract
):
    sorted_trace_lengths = sorted(trace_lengths)
    extracted_state_seqs = []
    extracted_events = []
    extracted_ep_durations = []
    extracted_actions = []
    for i in range(no_of_traces_to_extract):
        (length, index) = sorted_trace_lengths[i]
        extracted_state_seqs.append(states[index])
        extracted_events.append(events[index])
        extracted_actions.append(actions[index])
        extracted_ep_durations.append(length)
    return extracted_ep_durations, extracted_state_seqs, extracted_events, extracted_actions


def get_random_succ_traces(env, num_succ_traces, num_episodes):
    episode_durations, states_traversed, episode_events, actions_per_episodes = run_agent(
        env, num_episodes)

    return extract_shortest_succ_traces(
        episode_durations, states_traversed, episode_events, actions_per_episodes, num_succ_traces
    )


def get_random_succ_trace(env):
    ep_durs, states, events, actions = get_random_succ_traces(env, 1, 1)
    return ep_durs[0], states[0], events[0], actions[0]


# Given a list of state sequences, encodes every state in every sequence with the QBN given
def encode_state_seqs(qbn, state_seqs):
    encoded_seqs = []
    for state_seq in state_seqs:
        encoded_state_seq = encode_state_seq(qbn, state_seq)
        encoded_seqs.append(encoded_state_seq)
    return encoded_seqs

def encode_state_seq(qbn, state_seq):
    encoded_state_seq = []
    state_seq = list(map(lambda state: torch.tensor(state).float(), state_seq))
    for state in state_seq:
        encoded_state = qbn.encode(state)
        encoded_state_seq.append(encoded_state.detach().numpy())
    return encoded_state_seq

# Reduce the dimensionality of the data into two dimensions. This will help for visualisation of the KMeans clustering
def dim_reduction(states, use_tsne):
    principal_components = None
    if use_tsne:
        tsne_obj = TSNE(n_components=2)
        principal_components = tsne_obj.fit_transform(states)
    else:
        pca_obj = PCA(n_components=2)
        principal_components = pca_obj.fit_transform(states)
    return principal_components


# Plots the KMeans clusters
def visualise_training_clustering(labels, state_seq, no_of_clusters, plot_title):
    print("Visualising the clustering that is done when training the cluster objects")
    principal_components = dim_reduction(state_seq, use_tsne=False)
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=["PC1", "PC2"]
    )
    for i in range(no_of_clusters):
        filtered_df = pca_df[labels == i]
        plt.scatter(
            filtered_df["PC1"],
            filtered_df["PC2"],
            label=i,
        )
    plt.title(plot_title)
    plt.legend()
    plt.show()


# Performs KMeans clustering on the state sequences and plots the resulting data
def kmeans_clustering(state_seqs, no_of_clusters, plot_title):
    conc_state_seqs = np.concatenate(state_seqs)
    kmeans = KMeans(n_clusters=no_of_clusters)
    print("Fitting the states with the KMeans object")
    kmeans_obj = kmeans.fit(np.array(conc_state_seqs, dtype=np.double))
    # visualise_training_clustering(kmeans_obj.labels_, conc_state_seqs, no_of_clusters, plot_title)
    return kmeans_obj


# Calculates the cosine similarity score between two vectors and records the indices of states that are similar enough
def extract_sim_states(state_seqs, sim_threshold):
    similar_states_indices = [
        [[] for _ in range(len(state_seqs[x]))] for x in range(len(state_seqs))
    ]
    for i in range(len(state_seqs)):
        for j in range(len(state_seqs)):
            state_seq_i = state_seqs[i]
            state_seq_j = state_seqs[j]
            if state_seq_i != state_seq_j:
                cos_sims = cosine_similarity(state_seq_i, state_seq_j)
                for x in range(len(state_seq_i)):
                    for y in range(len(state_seq_j)):
                        cos_sim = cos_sims[x][y]
                        if cos_sim >= sim_threshold:
                            similar_states_indices[i][x].append(
                                (j, y, cos_sim))
    return similar_states_indices


# Extracts event labels by comparing state vectors across different successful traces
def extract_events_from_pairwise_comp(state_seqs):
    sim_threshold = 0.82
    sim_states_indices = extract_sim_states(state_seqs, sim_threshold)
    print(
        "Indices of similar states. Similarity score threshold: {}".format(
            sim_threshold
        )
    )
    for i in range(len(state_seqs)):
        for j in range(len(sim_states_indices[i])):
            print("Vectors similar to the {}th state in the {}th trace:".format(j, i))
            print(sim_states_indices[i][j])
            print("-------------")
        print("----------------------------------------------")
    tl.plot_sim_states_freq(sim_states_indices)
    # TODO: Change approach of extracting labels from similar states (this is naive)
    event_labels = [set() for _ in range(len(np.concatenate(state_seqs)))]
    return event_labels


def train_clustering(state_seqs, no_of_clusters, use_velocities, activation, encode_states=False):
    # cluster_obj_dir = "./trainedClusterObjs"
    # cluster_obj_qbn_loc = cluster_obj_dir + "/kmeans_qbn.pkl"
    # cluster_obj_no_qbn_loc = cluster_obj_dir + "/kmeans_no_qbn.pkl"

    qbn = None
    if encode_states: 
        qbn_filename = find_qbn_filename(activation, use_velocities)
        activation_class = convert_activation(activation)
        vec_dim = 52 if use_velocities else 28
        qbn = tl.loadSavedQBN("../../qbnTraining/trainedQBN/" + qbn_filename, vec_dim, activation_class)

        # Encode every state (tensor object) in every sequence with the QBN
        print("Using a loaded QBN to encode states")
        state_seqs = encode_state_seqs(qbn, state_seqs)
    
    plot_title = "KMeans clustering with encoding" if encode_states else "KMeans clustering without encoding"
    kmeans_obj = kmeans_clustering(state_seqs, no_of_clusters, plot_title)  
    
    return kmeans_obj, qbn
    # print("Save trained KMeans objects in pickle objects")
    # pickle.dump(kmeans_obj, open(cluster_obj_no_qbn_loc, "wb"))
    # pickle.dump(kmeans_obj_qbn, open(cluster_obj_qbn_loc, "wb"))


if __name__ == "__main__":
    env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "environment_seed": 0},
    )
    train_clustering(env=env, no_of_events=2, num_succ_traces=50, num_episodes=500)
