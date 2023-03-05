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
import torch
import random
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
def trainingLoop(model, obs_training_data):
    mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    optimizer = optim.Adam(qbn.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
    quantised_vectors = []
    total_batches = math.ceil(model.training_set_size / model.batch_size)
    
    # QBN training loop
    print("Beginning training of QBN")
    for epoch in range(model.epochs):
        model.train()
        loss_values = []
        total_loss = 0
        random.shuffle(obs_training_data)
        for b_i in range(total_batches):
            batch_input = obs_training_data[(b_i * model.batch_size) : (b_i * model.batch_size) + model.batch_size]
            batch_input = torch.FloatTensor(batch_input)
            batch_target = Variable(batch_input)
            batch_input = Variable(batch_input, requires_grad=True)

            if (torch.cuda.is_available()):
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            quantised_vector, feature_reconstruction = model.forward(batch_input)
            quantised_vectors.append(quantised_vector)

            optimizer.zero_grad()
            total_loss += loss.item()
            loss = mse_loss(feature_reconstruction, batch_target)
            loss.backward()
            # Loss value rounded to 2dp before adding to record of loss values
            # loss_values.append(round(loss.item(), 2))
            optimizer.step()

        print('Epoch: %d, Average Loss: %f' % (epoch, total_loss / total_batches))
    return model

# Evaluates the model (given as argument) after it has been trained
def evaluateQBN(model, test_data, batch_size):
    total_test_batches = math.ceil(len(test_data) / batch_size)
    loss_total = 0
    with torch.no_grad():
      for b_i in range(total_test_batches):
        batch_input = test_data[(b_i * batch_size) : (b_i * batch_size) + batch_size]
        batch_input = Variable(torch.FloatTensor(batch_input))
        batch_target = Variable(torch.FloatTensor(batch_input))
        if torch.cuda.is_available():
          batch_target, batch_input = batch_target.cuda(), batch_input.cuda()
        
        encoding, reconstruction = model(batch_input)

        mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
        loss = mse_loss(reconstruction, batch_target)
        loss_total += loss

        print("Input:")
        print(batch_input)
        print("Encoding:")
        print(encoding)
        print("Decoding:")
        print(reconstruction)
        print("Loss: %f", loss)

    return loss_total / len(test_data)

if __name__ == '__main__':
    # args = tl.get_args() Not using this yet, will do once hyperparameters have been tuned
    # env = gym.make("gym_subgoal_automata:{}".format(args.env), params={"generation": "random", "environment_seed": args.env_seed})
    env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0", params={"generation": "random", "environment_seed": 0})

    # Hyperparameters
    quant_vector_dim = 100
    training_batch_size = 64
    batch_size = 64
    learning_rate = 1e-5
    # learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    weight_decay = 1e-2
    # weight_decays = [0, 1e2, 1e5, 1e8]
    epochs = 50
    # epochs_array = [20, 50, 100, 200]
    training_set_size = 2000
    # training_set_sizes = [500, 1000, 2000, 5000]
    testing_set_size = 1000

    # Generate training dataset
    print("Beginning to generate training and testing data")
    obs_training_data = tl.generate_training_data(env=env, size_of_dataset=training_set_size)
    obs_testing_data = tl.generate_training_data(env=env, size_of_dataset=testing_set_size)

    print("Finished generating training and testing data")

    qbn = QuantisedBottleneckNetwork(quant_vector_dim, training_batch_size, learning_rate, weight_decay, epochs, training_set_size)

    qbn = trainingLoop(qbn, obs_training_data)

    average_loss = evaluateQBN(qbn, obs_testing_data, batch_size)
    print("Average Loss: {}".format(average_loss))