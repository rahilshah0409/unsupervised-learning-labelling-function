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

if __name__ == '__main__':
    # args = tl.get_args() Not using this yet, will do once hyperparameters have been tuned
    # env = gym.make("gym_subgoal_automata:{}".format(args.env), params={"generation": "random", "environment_seed": args.env_seed})
    env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0", params={"generation": "random", "environment_seed": 0})

    # Generate training dataset
    print("Beginning to generate training and testing data")
    obs_training_data = tl.generate_training_data(env=env, size_of_dataset=2000)
    obs_testing_data = tl.generate_training_data(env=env, size_of_dataset=1000)

    print("Finished generating training and testing data")

    # Fixed parameters
    quant_vector_dim = 6
    batch_size = 16

    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    weight_decays = [0, 1e2, 1e5, 1e8]
    epochs_array = [20, 50, 100, 200]
    training_set_sizes = [500, 1000, 2000, 5000]

    qbn = QuantisedBottleneckNetwork(quant_vector_dim=6)
    best_params, best_estimator = QBNHyperParameterSearch(qbn, obs_training_data, obs_training_data)

    print("Score on test data: ", best_estimator.score(obs_testing_data, obs_testing_data))
    print(best_params)

    new_qbn = QuantisedBottleneckNetwork(best_params['quant_vector_dim'], best_params['batch_size'], best_params['learning_rate'], best_params['weight_decay'], best_params['epochs'], best_params['training_set_size'])

    new_qbn.fit(obs_training_data, obs_training_data)
    torch.save(new_qbn, "best_model.pt")