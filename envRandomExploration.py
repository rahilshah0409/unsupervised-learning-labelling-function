import math
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import torch
import gym
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns

from qbn import QuantisedBottleneckNetwork

# Defines a policy for an agent to choose actions at each timestep. Initially random
def choose_action():
    NUM_ACTIONS = 5
    action_probs = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
    action = random.choices(range(NUM_ACTIONS), weights=action_probs, k=1)[0]
    return action

# Extracts the n shortest successful traces, where n is no_of_traces_to_extract
def extractShortestSuccessfulTraces(trace_lengths, states, events, no_of_traces_to_extract):
    sorted_trace_lengths = sorted(trace_lengths)
    print(sorted_trace_lengths)
    extracted_state_seqs = []
    extracted_events = []
    for i in range(no_of_traces_to_extract):
        (_, index) = sorted_trace_lengths[i]
        extracted_state_seqs.append(states[index])
        extracted_events.append(events[index])
    return extracted_state_seqs, extracted_events

# Runs an agent in the given environment for a given number of episodes, keeping track of the states traversed and events observed at every time step
def runAgentWithPolicy(env, num_episodes):
    episode_durations = []
    states_traversed = []
    events_per_episode = []
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))
        
        initial_state = env.reset()
        state = torch.tensor(initial_state).float()
        states = []
        states.append(state)
        events_observed = [set()]
        done, terminated, t = False, False, 1
        while not (done or terminated):
            action = choose_action()
            next_state, _, done, observations = env.step(action)
            next_state = torch.tensor(next_state).reshape(-1).float()
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

# Given a list of state sequences, encodes every state in every sequence with the QBN given
def encodeStateSeqs(qbn, state_seqs):
    encoded_seqs = []
    for state_seq in state_seqs:
        encoded_state_seq = []
        for state in state_seq:
            encoded_state = qbn.encode(state)
            encoded_state_seq.append(encoded_state.detach().numpy())        
        encoded_seqs.append(encoded_state_seq)
    return encoded_seqs

# Calculates the cosine similarity score between any pair of state vectors
# INCOMPLETE
def extractSimilarStates(state_seqs, sim_threshold):
    for i in range(len(state_seqs)):
        cos_sims_per_j = [[[] for y in range(len(state_seqs[x]))] for x in range(len(state_seqs))]
        for j in range(len(state_seqs)):
            if (i != j):
                state_seq_i = state_seqs[i]
                state_seq_j = state_seqs[j]
                cos_sims = []
                for state_i in range(state_seq_i):
                    cos_sims_i = []
                    for state_j in range(state_seq_j):
                        cos_sim = cosine_similarity(state_i, state_j)
                        cos_sims_i.append(cos_sim)
                    cos_sims.append(cos_sims_i)
                cos_sims_per_j[j] = cos_sims

# Calculates the cosine similarity score between two vectors and records the indices of states that are similar enough (where enough is determined by a threshold score provided as input)
def extractSimilarStates(state_seqs, sim_threshold):
    similar_states_indices = []
    for i in range(len(state_seqs)):
        for j in range(len(state_seqs)):
            state_seq_i = state_seqs[i]
            state_seq_j = state_seqs[j]
            print(len(state_seq_i))
            print(len(state_seq_j))
            if (state_seq_i != state_seq_j):
                for x in range(len(state_seq_i)):
                    for y in range(len(state_seq_j)):
                        state_x = np.array([state_seq_i[x]])
                        state_y = np.array([state_seq_j[y]])
                        cos_sim = cosine_similarity(state_x, state_y)
                        if (cos_sim >= sim_threshold):
                            similar_states_indices.append(((i, x), (j, y)))
    return similar_states_indices

# Uses principal component analysis to reduce the dimensionality of the data into two dimensions. This will help for visualisation of the KMeans clustering
def datasetDimReduction(state_seqs):
    # print(state_seqs)
    flattened_states = np.concatenate(state_seqs)
    pca2 = PCA(n_components=2)
    print(flattened_states)
    principal_components = pca2.fit_transform(flattened_states)
    return principal_components

# Plots the KMeans clusters
def plotKMeansClustering(df, labels, no_of_clusters):
    print("Labels (corresponding to different clusters)")
    print(labels)
    # print(df)
    for i in range(no_of_clusters):
        filtered_df = df[labels == i]
        print("Filtered dataframe for label {}".format(i))
        print(filtered_df)
        plt.scatter(filtered_df['Principal component 1'], filtered_df['Principal component 2'], label = i)
    plt.legend()
    plt.show()

# Performs KMeans clustering on the state sequences and plots the resulting data
def kmeansClustering(state_seqs, no_of_clusters):
    principal_components = datasetDimReduction(state_seqs)
    pca_df = pd.DataFrame(data=principal_components, columns=['Principal component 1', 'Principal component 2'])
    kmeans = KMeans(n_clusters=no_of_clusters)
    labels = kmeans.fit_predict(pca_df)
    plotKMeansClustering(pca_df, labels, no_of_clusters)

if __name__ == '__main__':
    env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0", params={"generation": "random", "environment_seed": 0})
    trained_model_loc = "./trainedQBN/finalModel.pth"

    NUM_SUCC_TRACES = 2
    NUM_EPISODES = 10

    # Generate successful traces for the task
    episode_durations, states_traversed, episode_events = runAgentWithPolicy(env, NUM_EPISODES)

    # Get the shortest traces- they are the most relevant
    relevant_state_seqs, relevant_events = extractShortestSuccessfulTraces(episode_durations, states_traversed, episode_events, NUM_SUCC_TRACES)
    # print(relevant_state_seqs)

    # Load the QBN (trained through the program qbnTrainAndEval.py)
    input_vec_dim = 52
    quant_vector_dim = 100
    training_batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0
    epochs = 300
    training_set_size = 8192
    testing_set_size = 2048
    qbn = QuantisedBottleneckNetwork(input_vec_dim, quant_vector_dim, training_batch_size, learning_rate, weight_decay, epochs, training_set_size)
    qbn.load_state_dict(torch.load(trained_model_loc))
    print("QBN loaded")

    # Encoded every state in every sequence with the QBN
    print("Using QBN to encode states")
    encoded_seqs = encodeStateSeqs(qbn, relevant_state_seqs)

    # Extract labels from latent vectors through pairwise comparison with cosine similarity
    # print("Applying cosine similarity to each pair")
    # sim_threshold = 0.82
    # sim_states_indices = extractSimilarStates(encoded_seqs, sim_threshold)
    # TODO: Extract labels from similar states
    # print("Indices of similar states. Similarity score threshold: {}".format(sim_threshold))
    # print(sim_states_indices)
    # print(len(sim_states_indices))

    # Alternatively, extract labels from latent vectors with k-means clustering
    print("Clustering the state sequences with KMeans and PCA")
    no_of_events = 2
    kmeansClustering(encoded_seqs, 2 ** no_of_events)

    # TODO: Perform evaluation of extracted labels 
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

