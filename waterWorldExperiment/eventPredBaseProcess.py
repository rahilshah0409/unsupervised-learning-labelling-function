import os
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
import tools as tl
from qbn import QuantisedBottleneckNetwork


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
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))

        state = env.reset()
        states = []
        states.append(state)
        events_observed = [set()]
        done, terminated, t = False, False, 1
        while not (done or terminated):
            action = choose_action()
            next_state, _, done, observations = env.step(action)
            state = next_state
            states.append(state)
            t += 1

            # if (observations != set()):
            events_observed.append(observations)
            if done:
                episode_durations.append((t, ep))

        states_traversed.append(states)
        events_per_episode.append(events_observed)
    return episode_durations, states_traversed, events_per_episode


# Extracts the n shortest successful traces, where n is no_of_traces_to_extract
def extract_shortest_succ_traces(
        trace_lengths, states, events, no_of_traces_to_extract
):
    sorted_trace_lengths = sorted(trace_lengths)
    extracted_state_seqs = []
    extracted_events = []
    extracted_ep_durations = []
    for i in range(no_of_traces_to_extract):
        (length, index) = sorted_trace_lengths[i]
        extracted_state_seqs.append(states[index])
        extracted_events.append(events[index])
        extracted_ep_durations.append(length)
    return extracted_ep_durations, extracted_state_seqs, extracted_events


# Given a list of state sequences, encodes every state in every sequence with the QBN given
def encode_state_seqs(qbn, state_seqs):
    encoded_seqs = []
    state_seqs = list(
        map(
            lambda seq: list(
                map(lambda state: torch.tensor(state).float(), seq)),
            state_seqs,
        )
    )
    for state_seq in state_seqs:
        encoded_state_seq = []
        for state in state_seq:
            encoded_state = qbn.encode(state)
            encoded_state_seq.append(encoded_state.detach().numpy())
        encoded_seqs.append(encoded_state_seq)
    return encoded_seqs


# Reduce the dimensionality of the data into two dimensions. This will help for visualisation of the KMeans clustering
def dataset_dim_reduction(states, use_tsne):
    principal_components = None
    if use_tsne:
        tsne = TSNE(n_components=2)
        principal_components = tsne.fit_transform(states)
    else:
        pca2 = PCA(n_components=2)
        principal_components = pca2.fit_transform(states)
    return principal_components


# Plots the KMeans clusters
def plot_kmeans_clustering(df, labels, no_of_clusters):
    for i in range(no_of_clusters):
        filtered_df = df[labels == i]
        plt.scatter(
            filtered_df["Principal component 1"],
            filtered_df["Principal component 2"],
            label=i,
        )
    plt.legend()
    plt.show()


# Performs KMeans clustering on the state sequences and plots the resulting data
def kmeans_clustering(state_seqs, no_of_clusters):
    principal_components = dataset_dim_reduction(states=state_seqs, use_tsne=False)
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=["Principal component 1", "Principal component 2"],
    )
    kmeans = KMeans(n_clusters=no_of_clusters)
    labels = kmeans.fit_predict(pca_df)
    plot_kmeans_clustering(pca_df, labels, no_of_clusters)
    return labels


# Given the cluster that each event is assigned to, extract an event label from them
# TODO: Change this approach (currently naive)
def extract_labels_from_clusters(cluster_labels):
    event_labels = []
    hardcoded_mapping = {"0": set(), "1": {"r"}, "2": {"b"}, "3": {"b", "r"}}
    for i in range(len(cluster_labels)):
        cluster_label_str = str(cluster_labels[i])
        event_labels.append(hardcoded_mapping[cluster_label_str])
    return event_labels


# Extract labels for events with a clustering method
def extract_events_from_clustering(state_seqs):
    print("Clustering the state sequences with KMeans and PCA")
    conc_state_seqs = np.concatenate(state_seqs)
    no_of_events = 2
    cluster_labels = kmeans_clustering(conc_state_seqs, 2 ** no_of_events)
    return cluster_labels
    # print("Extracting labels from the clusters")
    # event_labels = extract_labels_from_clusters(cluster_labels)
    # return event_labels


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
    print(len(sim_states_indices))
    # TODO: Change approach of extracting labels from similar states (this is naive)
    event_labels = [set() for _ in range(len(np.concatenate(state_seqs)))]
    return event_labels


# Extracts event labels of the given state sequences, either by pairwise comparison or clustering
def extract_events(state_seqs, pairwise_comp):
    return (
        extract_events_from_pairwise_comp(state_seqs)
        if pairwise_comp
        else extract_events_from_clustering(state_seqs)
    )


def runEventPrediction(env, num_succ_traces, num_episodes, use_pairwise_comp):
    trained_model_loc = "./trainedQBN/finalModel.pth"

    # Load the QBN (trained through the program qbnTrainAndEval.py)
    input_vec_dim = 52
    quant_vector_dim = 100
    training_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0
    epochs = 300
    training_set_size = 8192
    qbn = QuantisedBottleneckNetwork(
        input_vec_dim,
        quant_vector_dim,
        training_batch_size,
        learning_rate,
        weight_decay,
        epochs,
        training_set_size,
    )
    qbn.load_state_dict(torch.load(trained_model_loc))
    print("QBN loaded")

    # Generate successful traces for the task
    episode_durations, states_traversed, episode_events = run_agent(
        env, num_episodes)

    # Get the shortest traces- they are the most relevant
    (
        shortest_ep_durations,
        relevant_state_seqs,
        relevant_events,
    ) = extract_shortest_succ_traces(
        episode_durations, states_traversed, episode_events, num_succ_traces
    )

    # Encoded every state (tensor object) in every sequence with the QBN
    print("Using QBN to encode states")
    encoded_seqs = encode_state_seqs(qbn, relevant_state_seqs)

    # Alternatively, extract labels from latent vectors with k-means clustering (cluster labels)
    event_labels = extract_events(state_seqs=encoded_seqs, pairwise_comp=use_pairwise_comp)

    conc_relevant_events = np.concatenate(relevant_events)
    tl.plot_events_pred_events_from_env_dist(event_labels, conc_relevant_events, shortest_ep_durations)
    precision_scores, recall_scores = tl.compare_changes_in_events(event_labels, conc_relevant_events, shortest_ep_durations)
    print(precision_scores)
    print(recall_scores)

    # Perform evaluation of extracted labels
    # conc_relevant_events = np.concatenate(relevant_events)
    # correct_labels = [
    #     l1 for l1, l2 in zip(event_labels, conc_relevant_events) if l1 == l2
    # ]
    # accuracy = len(correct_labels) / len(conc_relevant_events)
    # print("Accuracy of predicted mapping of event labels is {}".format(accuracy))

    # return accuracy


if __name__ == "__main__":
    # env = gym.make(
    #     "gym_subgoal_automata:WaterWorldRed-v0",
    #     params={"generation": "random", "environment_seed": 0},
    # )
    env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "environment_seed": 0},
    )
    runEventPrediction(env=env, num_succ_traces=2, num_episodes=10, use_pairwise_comp=False)
