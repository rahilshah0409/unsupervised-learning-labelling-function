import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import torch
import gym
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Defines a policy for an agent to choose actions at each timestep. Initially random
def choose_action():
    NUM_ACTIONS = 4
    action_probs = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
    action = random.choices(range(NUM_ACTIONS), weights=action_probs, k=1)[0]
    return action

# Runs an agent in the given environment for a given number of episodes, keeping track of the states traversed and events observed at every time step
def run_agent(env, num_episodes):
    episode_durations = []
    states_traversed = []
    events_per_episode = []
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))
        states = []
        state = env.reset()
        states.append(state)
        # How would you get the event that is observed in the initial state?
        events_observed = [set()]
        done, t = False, 1
        while not done:
            action = choose_action()
            next_state, reward, done, observations = env.step(action)
            state = next_state
            states.append(state)
            t += 1

            events_observed.append(observations)
            if done:
                episode_durations.append((t, ep, reward == 1.0))

        states_traversed.append(states)
        events_per_episode.append(events_observed)
    return episode_durations, states_traversed, events_per_episode

# Extracts the n shortest successful traces, where n is no_of_traces_to_extract
def get_relevant_succ_traces(trace_lengths, states, events, no_of_traces_to_extract):
    succ_traces = list(filter(lambda trace_info: trace_info[2], trace_lengths))
    sorted_trace_lengths = sorted(succ_traces)
    relevant_state_seqs = []
    relevant_events = []
    relevant_ep_lens = []
    if no_of_traces_to_extract < len(sorted_trace_lengths):
        for i in range(no_of_traces_to_extract):
            (length, index, _) = sorted_trace_lengths[i]
            relevant_state_seqs.append(states[index])
            relevant_events.append(events[index])
            relevant_ep_lens.append(length)
    else:
        print("There is not enough traces! There is only {} traces".format(
            len(sorted_trace_lengths)))
    return relevant_ep_lens, relevant_state_seqs, relevant_events

# Uses principal component analysis to reduce the dimensionality of the data into two dimensions. This will help for visualisation of the KMeans clustering
def dataset_dim_reduction(states):
    pca2 = PCA(n_components=2)
    principal_components = pca2.fit_transform(states)
    return principal_components

# Plots the KMeans clusters
def plot_kmeans_clustering(df, labels, no_of_clusters):
    for i in range(no_of_clusters):
        filtered_df = df[labels == i]
        plt.scatter(filtered_df['Principal component 1'],
                    filtered_df['Principal component 2'], label=i)
    plt.legend()
    plt.show()

# Performs KMeans clustering on the state sequences and plots the resulting data
def kmeans_clustering(state_seqs, no_of_clusters):
    principal_components = dataset_dim_reduction(state_seqs)
    pca_df = pd.DataFrame(data=principal_components, columns=[
                          'Principal component 1', 'Principal component 2'])
    kmeans = KMeans(n_clusters=no_of_clusters)
    labels = kmeans.fit_predict(pca_df)
    # plotKMeansClustering(pca_df, labels, no_of_clusters)
    return labels

# Given the cluster that each event is assigned to, extract an event label from them
# TODO: Change this approach (currently naive)
def extract_labels_from_clusters(cluster_labels):
    event_labels = []
    hardcoded_mapping = {'0': set(), '1': {'r'}, '2': {'b'}, '3': {'b', 'r'}}
    for i in range(len(cluster_labels)):
        cluster_label_str = str(cluster_labels[i])
        event_labels.append(hardcoded_mapping[cluster_label_str])
    return event_labels

def plot_cluster_labels_over_traces(state_seqs, cluster_labels):
    succ_trace_index = 0
    state_index_in_trace = 0
    cluster_label_index = 0
    fig, axs = plt.subplots(NUM_SUCC_TRACES)
    fig.suptitle("Cluster labels for different traces")

    y_axis = []
    while succ_trace_index < NUM_SUCC_TRACES:
        label = cluster_labels[cluster_label_index]
        y_axis.append(label)
        state_index_in_trace += 1
        cluster_label_index += 1
        if state_index_in_trace >= len(state_seqs[succ_trace_index]):
            x_axis = range(state_index_in_trace)
            axs[succ_trace_index].plot(x_axis, y_axis)
            axs[succ_trace_index].set_title(
                "Trace {}".format(succ_trace_index))
            axs[succ_trace_index].set(
                xlabel="State index", ylabel="Cluster labels")
            succ_trace_index += 1
            state_index_in_trace = 0
            y_axis = []

    plt.show()

def extract_events_from_clustering(state_seqs):
    print("Clustering the state sequences with KMeans and PCA")
    conc_state_seqs = np.concatenate(state_seqs)
    no_of_events = 2
    cluster_labels = kmeans_clustering(conc_state_seqs, 2 ** no_of_events)
    print(cluster_labels)
    plot_cluster_labels_over_traces(state_seqs, cluster_labels)
    print("Extracting labels from the clusters")
    event_labels = extract_labels_from_clusters(cluster_labels)
    return event_labels

# Calculates the cosine similarity score between two vectors and records the indices of states that are similar enough (where enough is determined by a threshold score provided as input)
def extract_sim_states(state_seqs, sim_threshold):
    similar_states_indices = [
        [[] for _ in range(len(state_seqs[x]))] for x in range(len(state_seqs))]
    for i in range(len(state_seqs)):
        for j in range(len(state_seqs)):
            state_seq_i = state_seqs[i]
            state_seq_j = state_seqs[j]
            print(len(state_seq_i))
            print(len(state_seq_j))
            if (state_seq_i != state_seq_j):
                cos_sims = cosine_similarity(state_seq_i, state_seq_j)
                for x in range(len(state_seq_i)):
                    for y in range(len(state_seq_j)):
                        cos_sim = cos_sims[x][y]
                        if (cos_sim >= sim_threshold):
                            similar_states_indices[i][x].append(
                                (j, y, cos_sim))
    return similar_states_indices

# Extracts event labels by comparing state vectors across different successful traces
def extract_events_from_pairwise_comp(state_seqs):
    sim_threshold = 0.82
    sim_states_indices = extract_sim_states(state_seqs, sim_threshold)
    print("Indices of similar states. Similarity score threshold: {}".format(
        sim_threshold))
    for i in range(NUM_SUCC_TRACES):
        for j in range(len(sim_states_indices[i])):
            print("Vectors similar to the {}th state in the {}th trace:".format(j, i))
            print(sim_states_indices[i][j])
            print("-------------")
        print("----------------------------------------------")
    print(len(sim_states_indices))
    # TODO: Change approach of extracting labels from similar states (this is naive)
    event_labels = [set() for _ in range(len(np.concatenate(state_seqs)))]
    return event_labels

# Extracts event labels of the given state sequences, either by pairwise comparison or clustering
def extract_events(state_seqs, pairwise_comp):
    return extract_events_from_pairwise_comp(state_seqs) if pairwise_comp else extract_events_from_clustering(state_seqs)


if __name__ == '__main__':
    env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
                   params={"generation": "random", "environment_seed": 0})

    NUM_SUCC_TRACES = 2
    NUM_EPISODES = 100

    # Generate successful traces for the task
    episode_durations, states_traversed, episode_events = run_agent(
        env, NUM_EPISODES)

    # Get the shortest traces- they are the most relevant
    shortest_ep_durations, relevant_state_seqs, relevant_events = get_relevant_succ_traces(
        episode_durations, states_traversed, episode_events, NUM_SUCC_TRACES)

    print(shortest_ep_durations)
    print(relevant_state_seqs)
    print(relevant_events)

    # Transforming each state from a tensor object to np.ndarray
    # relevant_state_seqs = list(map(lambda seq: list(map(lambda state: state.detach().numpy(), seq)), relevant_state_seqs))

    # # Alternatively, extract labels from latent vectors with k-means clustering
    # event_labels = extract_events(state_seqs=encoded_seqs, pairwise_comp=False)

    # # Perform evaluation of extracted labels
    # conc_relevant_events = np.concatenate(relevant_events)
    # correct_labels = [l1 for l1, l2 in zip(event_labels, conc_relevant_events) if l1 == l2]
    # accuracy = len(correct_labels) / len(conc_relevant_events)
    # print("Accuracy of predicted mapping of event labels is {}".format(accuracy))

    # for ((i, x), (j, y)) in sim_states_indices:
    #     # print("({}, {}), ({}, {})".format(i, x, j, y))
    #     events_observed_1 = relevant_events[i][x]
    #     events_observed_2 = relevant_events[j][y]
    #     if (events_observed_1 != set() and events_observed_2 != set()):
    #         print("({}, {}), ({}, {})".format(i, x, j, y))
    #         print("Events observed 1")
    #         print(events_observed_1)
    #         print("Events observed 2")
    #         print(events_observed_2)
