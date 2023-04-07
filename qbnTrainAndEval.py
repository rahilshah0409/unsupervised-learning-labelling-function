# Python program that allows for experimentation with the Water World environment

import math
import sys
import time

from sklearn.model_selection import RandomizedSearchCV
from qbn import QuantisedBottleneckNetwork
import gym
import tools as tl
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
  search = RandomizedSearchCV(estimator=model, param_distributions=potential_params, n_iter=25, cv=5, scoring=None)
  search.fit(x, y)
  return search.best_params_, search.best_estimator_

# Method that carries out the training loop, hyperparameters chosen are in the model object itself
def trainingLoop(model, train_data, test_data, test_batch_size):
    mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    optimizer = optim.Adam(qbn.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
    quantised_vectors = []
    total_train_batches = math.ceil(model.training_set_size / model.batch_size)
    epoch_train_losses = []
    epoch_test_losses = []
    
    # QBN training loop
    print("Beginning training of QBN")
    for epoch in range(model.epochs):
        model.train()
        total_train_loss = 0
        random.shuffle(train_data)
        for b_i in range(total_train_batches):
            batch_input = train_data[(b_i * model.batch_size) : (b_i * model.batch_size) + model.batch_size]
            batch_input = torch.FloatTensor(np.array(batch_input))
            batch_target = Variable(batch_input)
            batch_input = Variable(batch_input, requires_grad=True)

            if (torch.cuda.is_available()):
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            quantised_vector, feature_reconstruction = model.forward(batch_input)
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

        average_test_loss = round(evaluateQBN(model, test_data, test_batch_size), 5)
        epoch_test_losses.append(average_test_loss)

        print('Epoch: {}, Training loss: {}, Test loss: {}'.format(epoch, average_loss, average_test_loss))
        # print('Epoch: %d, Average Loss: %f' % (epoch, total_loss / total_batches))

    epoch_loss_dict = {'title': 'Loss vs Epoch', 
                        'train_data': epoch_train_losses, 
                        'test_data': epoch_test_losses,
                        'y_label': 'Loss(' + str(min(epoch_train_losses)) + ')',
                        'x_label': 'Epoch',
                        'filename': 'loss_vs_epochs.png'}
    
    epoch_loss_dicts = [epoch_loss_dict]
    
    tl.plot_data(epoch_loss_dicts, "results/")
    return model

# Evaluates the model (given as argument) after it has been trained
def evaluateQBN(model, test_data, batch_size):
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

        # print("Input:")
        # print(batch_input)
        # print("Encoding:")
        # print(encoding)
        # print("Decoding:")
        # print(reconstruction)
        # print("Loss: %f", loss)

    return loss_total / total_test_batches

# Helper method that prints out datapoints in the dataset given as input
def inspectData(data):
  for i in range(100):
    print(data[i])

if __name__ == '__main__':
    # args = tl.get_args() Not using this yet, will do once hyperparameters have been tuned
    # env = gym.make("gym_subgoal_automata:{}".format(args.env), params={"generation": "random", "environment_seed": args.env_seed})
    env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0", params={"generation": "random", "environment_seed": 0})
    trained_model_loc = "./trainedQBN/finalModel.pth"

    input_vec_dim = 52

    # Hyperparameters
    quant_vector_dim = 100
    training_batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0
    epochs = 300
    training_set_size = 8192
    testing_set_size = 2048

    # Generate training dataset
    print("Beginning to generate training and testing data")

    obs_training_data = tl.generate_train_data_rand_init(env=env, dataset_size=training_set_size)
    obs_testing_data = tl.generate_train_data_rand_init(env=env, dataset_size=testing_set_size)

    print("Finished generating training and testing data")

    # inspectData(obs_training_data)

    qbn = QuantisedBottleneckNetwork(input_vec_dim, quant_vector_dim, training_batch_size, learning_rate, weight_decay, epochs, training_set_size)

    print("Training the QBN now")
    qbn = trainingLoop(qbn, obs_training_data, obs_testing_data, test_batch_size)
    print("Finished training the QBN")

    torch.save(qbn.state_dict(), trained_model_loc)

    average_loss = evaluateQBN(qbn, obs_testing_data, test_batch_size)
    print("Average Loss: {}".format(average_loss))