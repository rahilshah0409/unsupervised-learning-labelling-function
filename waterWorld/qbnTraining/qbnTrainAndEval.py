# Python program that allows for experimentation with the Water World environment

import math
from sklearn.model_selection import RandomizedSearchCV
from qbn import QuantisedBottleneckNetwork
import gym
import sys
from waterWorld.qbnTraining.quantisationMethods import BinarySigmoid
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/AEExperiment/AEExperiment")
import waterWorld.utils.tools as tl
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import torch
import random
import numpy as np
from torch.autograd import Variable
from scipy.stats import uniform

# Ignoring this method for now


def QBNHyperParameterSearch(model, x, y):
    potential_params = {
        "quant_vector_dim": list([4, 8, 16, 32, 64, 128]),
        "batch_size": list([4, 8, 16, 32, 64, 128]),
        "learning_rate": uniform(0.001, 0.1),
        "weight_decay": uniform(0, 0.2),
        "epochs": list(range(10, 100, 10)),
        "training_set_size": list([500, 1000, 2000, 3000])
    }
    search = RandomizedSearchCV(
        estimator=model, param_distributions=potential_params, n_iter=25, cv=5, scoring=None)
    search.fit(x, y)
    return search.best_params_, search.best_estimator_

# Method that carries out the training loop, hyperparameters chosen are in the model object itself


def train_loop(qbn, train_data, test_data, test_batch_size):
    mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    optimizer = optim.Adam(
        qbn.parameters(), lr=qbn.learning_rate, weight_decay=qbn.weight_decay)
    quantised_vectors = []
    total_train_batches = math.ceil(qbn.training_set_size / qbn.batch_size)
    epoch_train_losses = []
    epoch_test_losses = []

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

            quantised_vector, feature_reconstruction = qbn.forward(
                batch_input)
            quantised_vectors.append(quantised_vector)

            optimizer.zero_grad()
            loss = mse_loss(feature_reconstruction, batch_target)
            total_train_loss += loss.item()
            loss.backward()
            utils.clip_grad_norm_(qbn.parameters(), 5)
            # Loss value rounded to 2dp before adding to record of loss values
            # loss_values.append(round(loss.item(), 2))
            optimizer.step()

            # print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, b_i, loss.item()))

        average_loss = round(total_train_loss / total_train_batches, 5)
        epoch_train_losses.append(average_loss)

        average_test_loss = round(
            eval_qbn(qbn, test_data, test_batch_size), 5)
        epoch_test_losses.append(average_test_loss)

        print('Epoch: {}, Training loss: {}, Test loss: {}'.format(
            epoch, average_loss, average_test_loss))
        # print('Epoch: %d, Average Loss: %f' % (epoch, total_loss / total_batches))

    epoch_loss_dict = {'title': 'Loss vs Epoch',
                       'train_data': epoch_train_losses,
                       'test_data': epoch_test_losses,
                       'y_label': 'Loss(' + str(min(epoch_train_losses)) + ')',
                       'x_label': 'Epoch',
                       'filename': 'loss_vs_epochs.png'}

    epoch_loss_dicts = [epoch_loss_dict]

    tl.plot_data(epoch_loss_dicts, "trainingQBNResults/")
    return qbn

# Evaluates the model (given as argument) after it has been trained


def eval_qbn(model, test_data, batch_size):
    total_test_batches = math.ceil(len(test_data) / batch_size)
    loss_total = 0
    with torch.no_grad():
        for b_i in range(total_test_batches):
            batch_input = test_data[(b_i * batch_size) : (b_i * batch_size) + batch_size]
            batch_input = Variable(torch.FloatTensor(np.array(batch_input)))
            batch_target = Variable(torch.FloatTensor(batch_input))

            if torch.cuda.is_available():
                batch_target, batch_input = batch_target.cuda(), batch_input.cuda()

            encoding, reconstruction = model(batch_input)

            mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
            loss = mse_loss(reconstruction, batch_target)
            loss_total += loss.item()

    return loss_total / total_test_batches


def run_qbn_training(input_vec_dim, encoder_activation, train_data, test_data, trained_model_loc):
    # Hyperparameters
    quant_vector_dim = 100
    training_batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0
    epochs = 300
    training_set_size = 8192
    testing_set_size = 2048

    # Create and train the QBN
    qbn = QuantisedBottleneckNetwork(input_vec_dim, quant_vector_dim, training_batch_size,
                                     learning_rate, weight_decay, epochs, encoder_activation, training_set_size)
    print("Training the QBN now")
    qbn = train_loop(qbn, train_data, test_data, test_batch_size)
    print("Finished training the QBN")

    torch.save(qbn.state_dict(), trained_model_loc)


if __name__ == '__main__':
    # args = tl.get_args() Not using this yet, will do once hyperparameters have been tuned
    # env = gym.make("gym_subgoal_automata:{}".format(args.env), params={"generation": "random", "environment_seed": args.env_seed})
    normal_env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0",
                   params={"generation": "random", "environment_seed": 0})
    static_ball_env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0",
                   params={"generation": "random", "environment_seed": 0, "use_velocities": False})
    bin_sig_normal_loc = "./trainedQBN/binSigNormal.pth"
    bin_sig_static_loc = "./trainedQBN/binSigStatic.pth"
    tanh_normal_loc = "./trainedQBN/tanhNormal.pth"
    tanh_static_loc = "./trainedQBN/tanhStatic.pth"
    sig_normal_loc = "./trainedQBN/sigNormal.pth"
    sig_static_loc = "./trainedQBN/sigStatic.pth"
    normal_vec_dim = 52
    static_vec_dim = 28

    locs = [bin_sig_normal_loc, bin_sig_static_loc, tanh_normal_loc, tanh_static_loc, sig_normal_loc, sig_static_loc]
    input_dims = [normal_vec_dim, static_vec_dim, normal_vec_dim, static_vec_dim, normal_vec_dim, static_vec_dim]
    encoder_activations = [BinarySigmoid(), BinarySigmoid(), nn.Tanh(), nn.Tanh(), nn.Sigmoid(), nn.Sigmoid()]

    training_set_size = 8192
    testing_set_size = 2048
    print("Beginning to create the training and testing dataset")
    normal_train_data = tl.generate_train_data_rand_init(
        env=normal_env, dataset_size=training_set_size)
    normal_test_data = tl.generate_train_data_rand_init(
        env=normal_env, dataset_size=testing_set_size)
    static_train_data = tl.generate_train_data_rand_init(
        env=static_ball_env, dataset_size=training_set_size)
    static_test_data = tl.generate_train_data_rand_init(
        env=static_ball_env, dataset_size=training_set_size)
    print("Finished creating the training and testing dataset")

    train_data = [normal_train_data, static_train_data, normal_train_data, static_train_data, normal_train_data, static_train_data]
    test_data = [normal_test_data, static_test_data, normal_test_data, static_test_data, normal_test_data, static_test_data]

    for i in range(len(input_dims)):
        run_qbn_training(input_dims[i], encoder_activations[i], train_data[i], test_data[i], locs[i])

    # run_qbn_training(static_ball_env, input_vec_dim_static, trained_model_loc_static)

    # # Evaluate the model's performance
    # average_loss = eval_qbn(qbn, obs_testing_data, test_batch_size)
    # print("Average Loss: {}".format(average_loss))
