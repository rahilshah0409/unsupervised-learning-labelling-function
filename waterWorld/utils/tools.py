import math
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.utils as utils

from waterWorld.qbnTraining.qbn import QuantisedBottleneckNetwork

# Generates the dataset that is used for training the QBN for the feature vectors by randomly initialising the environment a given number of times
def generate_train_data_rand_init(env, dataset_size):
    obs_training_data = []
    for _ in range(dataset_size):
        rand_initial_state = env.reset()
        obs_training_data.append(rand_initial_state)
    return obs_training_data

# Generate the dataset used to evaluate the QBN
def generate_testing_data(env, dataset_size):
    return generate_train_data_rand_init(env, dataset_size)

# Generates the dataset used for QBN training by adding the states of successful traces when an agent follows a random policy
def generate_train_data_succ_traces(env, dataset_size):
    dataset = []

    while len(dataset) < dataset_size:
        initial_state = env.reset()
        state = torch.tensor(initial_state).float()
        dataset.append(state)

        done, terminated, t = False, False, 1

        while not (done or terminated):
            NUM_ACTIONS = 5
            action_probs = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
            action = random.choices(
                range(NUM_ACTIONS), weights=action_probs, k=1)[0]

            next_state, _, done, observations = env.step(action)

            next_state = torch.tensor(next_state).reshape(-1).float()
            state = next_state
            dataset.append(state)

            t += 1

    return dataset

# Gets the cmdline arguments
def get_args():
    """
    Arguments used to get input from command line.
    :return: given arguments in command line
    """
    parser = argparse.ArgumentParser(
        description='Training QBN for feature vectors')
    parser.add_argument('--generate_bn_data', action='store_true',
                        default=False, help='Generate Bottle-Neck Data')

    parser.add_argument('--train_qbn', action='store_true',
                        default=False, help='Train QBN')
    parser.add_argument('--test_qbn', action='store_true',
                        default=False, help='Test QBN')

    parser.add_argument('--quant_vector_dim', type=int,
                        help="Dimensions of discretized vector")

    parser.add_argument('--no_of_epochs', type=int,
                        default=400, help="No. of training episodes")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size used for training")

    parser.add_argument('--dataset_size', type=int, default=20,
                        help="Size of training set for Bottleneck Network training")
    parser.add_argument('--no_of_epochs', type=int,
                        default=100, help="No. of QBN training epochs")

    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='no cuda usage')
    parser.add_argument('--env', default="WaterWorldRedGreen-v0",
                        help="Name of the environment")
    parser.add_argument('--env_seed', type=int, default=0,
                        help="Seed for the environment")
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results")
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    return args

# Plots the information passed in as dict and saves the plot in directory
def plot_data(dict, directory):
    for x in dict:
        title = x['title']
        train_data = x['train_data']
        if len(train_data) == 1:
            plt.scatter([0], train_data)
        else:
            plt.plot(train_data, color='r', label='train')
        test_data = x['test_data']
        if len(test_data) == 1:
            plt.scatter([0], test_data)
        else:
            plt.plot(test_data, color='g', label='test')
        plt.grid(True)
        plt.title(title)
        plt.legend()
        plt.ylabel(x['y_label'])
        plt.xlabel(x['x_label'])
        plt.savefig(os.path.join(directory, x['filename']))
        plt.clf()

# Trains a given QBN with train_data
def trainQBN(qbn, train_data):
    mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    optimizer = optim.Adam(
        qbn.parameters(), lr=qbn.learning_rate, weight_decay=qbn.weight_decay)
    quantised_vectors = []
    total_train_batches = math.ceil(qbn.training_set_size / qbn.batch_size)
    epoch_train_losses = []

    # QBN training loop
    print("Beginning training of QBN")
    for epoch in range(qbn.epochs):
        qbn.train()
        total_train_loss = 0
        random.shuffle(train_data)
        for b_i in range(total_train_batches):
            batch_input = train_data[(
                b_i * qbn.batch_size): (b_i * qbn.batch_size) + qbn.batch_size]
            batch_input = torch.FloatTensor(np.array(batch_input))
            batch_target = Variable(batch_input)
            batch_input = Variable(batch_input, requires_grad=True)

            if (torch.cuda.is_available()):
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            quantised_vector, feature_reconstruction = qbn.forward(batch_input)
            quantised_vectors.append(quantised_vector)

            optimizer.zero_grad()
            loss = mse_loss(feature_reconstruction, batch_target)
            total_train_loss += loss.item()
            loss.backward()
            utils.clip_grad_norm_(qbn.parameters(), 5)
            optimizer.step()

        average_loss = round(total_train_loss / total_train_batches, 5)
        epoch_train_losses.append(average_loss)

        print('Epoch: {}, Training loss: {}'.format(epoch, average_loss))

    return qbn

# Plots graphs to compare how cluster labels are assigned to states in sequences vs the events that are observed in the same sequence
def plot_events_pred_events_from_env_dist(events_pred, events_from_env, ep_durations):
    succ_trace_index = 0
    state_index_in_trace = 0
    cluster_label_index = 0
    num_succ_traces = len(ep_durations)
    events_from_env = list(map(convert_obs_set_to_str, events_from_env))

    fig, axs = plt.subplots(num_succ_traces, 2)
    fig.suptitle("Cluster labels vs events from environment")

    y_axis_pred = []
    y_axis_from_env = []
    while succ_trace_index < len(ep_durations):
        label_pred = events_pred[cluster_label_index]
        event_from_env = events_from_env[cluster_label_index]
        y_axis_pred.append(label_pred)
        y_axis_from_env.append(event_from_env)
        state_index_in_trace += 1
        cluster_label_index += 1
        if state_index_in_trace >= ep_durations[succ_trace_index]:
            x_axis = range(state_index_in_trace)
            axs[succ_trace_index, 0].plot(x_axis, y_axis_pred, 'o')
            axs[succ_trace_index, 0].set(
                xlabel="State index", ylabel="Cluster labels")
            axs[succ_trace_index, 1].plot(x_axis, y_axis_from_env, 'o')
            axs[succ_trace_index, 1].set(
                xlabel="State index", ylabel="Event labels")
            succ_trace_index += 1
            state_index_in_trace = 0
            y_axis_pred = []
            y_axis_from_env = []

    plt.show()

# Visualised how cluster labels have been assigned to states in a trace vs the events that were actually observed in the same trace (one configuration)
def visualise_cluster_labels_vs_events(cluster_labels, plot_title, event_labels, ep_dur):
    event_labels = list(map(convert_obs_set_to_str, event_labels))

    fig, (axs1, axs2) = plt.subplots(1, 2)
    fig.suptitle(plot_title)

    x_axis = range(ep_dur)
    axs1.plot(x_axis, cluster_labels, 'o')
    axs1.set(
        xlabel="State index", ylabel="Cluster labels")
    axs2.plot(x_axis, event_labels, 'o')
    axs2.set(
        xlabel="State index", ylabel="Event labels")

    plt.show()


# Visualises how cluster labels have been assigned to states in a trace vs the events that were actually observed in the same trace (several configurations)
def visualise_cluster_labels_arr_vs_events(cluster_labels_arr, subplot_titles, event_labels, ep_dur):
    event_labels = list(map(convert_obs_set_to_str, event_labels))

    fig, axs = plt.subplots(len(cluster_labels_arr), 2)
    fig.suptitle(
        "Cluster labels vs ground event labels")

    x_axis = range(ep_dur)
    for i in range(len(cluster_labels_arr)):
        axs[i, 0].plot(x_axis, cluster_labels_arr[i], 'o')
        axs[i, 0].set(
            xlabel="State index", ylabel="Cluster labels", title=subplot_titles[i])
        axs[i, 1].plot(x_axis, event_labels, 'o')
        axs[i, 1].set(
            xlabel="State index", ylabel="Event labels")

    plt.show()


# Helper method that converts the observation set into a string
def convert_obs_set_to_str(obs):
    if obs == {'r', 'g'}:
        return "Both red and green"
    elif 'r' in obs:
        return "Red"
    elif 'g' in obs:
        return "Green"
    else:
        return "None"

# Calculates a numerical comparison between cluster jumps and event labels
def compare_changes_in_cluster_ids_vs_events(cluster_labels_arr, event_labels, ep_dur):
    event_labels = list(map(convert_obs_set_to_str, event_labels))
    last_cluster_labels = [labels[0] for labels in cluster_labels_arr]
    last_event_label = event_labels[0]

    changes_in_clusters = [[] for _ in range(len(cluster_labels_arr))]
    changes_in_env_events = []

    for i in range(1, ep_dur):
        curr_cluster_labels = [labels[i] for labels in cluster_labels_arr]
        curr_event_label = event_labels[i]
        if last_event_label != curr_event_label:
            changes_in_env_events.append(
                (i, last_event_label, curr_event_label))
        for j in range(len(cluster_labels_arr)):
            last = last_cluster_labels[j]
            curr = curr_cluster_labels[j]
            if last != curr:
                changes_in_clusters[j].append((i, last, curr))
        last_event_label = curr_event_label
        last_cluster_labels = curr_cluster_labels

    precision_scores = []
    recall_scores = []
    print("Changes in ground event labels")
    print(changes_in_env_events)
    for i in range(len(last_cluster_labels)):   
        print(changes_in_clusters[i])  
        precison, recall = precision_and_recall_calculator(
            changes_in_clusters[i], changes_in_env_events)
        precision_scores.append(precison)
        recall_scores.append(recall)

    return precision_scores, recall_scores


# Calculates the precision and recall of the changes in cluster labels and the changes in events taken from the environment
def precision_and_recall_calculator(changes_in_clusters, changes_in_env_events):
    cluster_indices = []
    if (changes_in_clusters != []):
        cluster_indices, _, _ = zip(*changes_in_clusters)
    env_e_indices = []
    if changes_in_env_events != []:
        env_e_indices, _, _ = zip(*changes_in_env_events)
    precise_changes = list(
        filter(lambda change: change[0] in env_e_indices, changes_in_clusters))
    precision = 0 if len(changes_in_clusters) == 0 else len(
        precise_changes) / len(changes_in_clusters)
    recall_changes = list(
        filter(lambda change: change[0] in cluster_indices, changes_in_env_events))
    recall = 0 if len(env_e_indices) == 0 else len(
        recall_changes) / len(env_e_indices)
    return precision, recall

# Plots how many states are similar to a given state in a sequence
def plot_sim_states_freq(sim_states_arr):
    fig, axs = plt.subplots(len(sim_states_arr))
    fig.suptitle("Distribution of similar states for each successful trace")

    for i in range(len(sim_states_arr)):
        axs[i].set(xlabel="State indices in trace {}".format(
            i), ylabel="Number of states similar to state at index i")
        yaxis = []
        for j in range(len(sim_states_arr[i])):
            yaxis.append(len(sim_states_arr[i][j]))
        axs[i].plot(range(len(sim_states_arr[i])), yaxis)

    plt.show()

# Loads an autoencoder (QBN) based on particular hyperparameters when given the location of the saved model
def loadSavedQBN(trained_model_loc, input_vec_dim, activation):
    # Load the QBN (trained through the program qbnTrainAndEval.py)
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
        activation,
        training_set_size,
    )
    qbn.load_state_dict(torch.load(trained_model_loc))
    return qbn
