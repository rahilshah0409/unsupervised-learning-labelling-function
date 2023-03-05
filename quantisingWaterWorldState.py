# This program intends to exploit the quantisation methods explained in "Learning Finite State Representations of Recurrent Policy Networks"

import gym
import random
import numpy as np
import torch
from qbn import QuantisedBottleneckNetwork

NO_OF_ACTIONS = 4

class QuantisingWaterWorldState():

  def __init__(self, env, training_set_size, no_of_training_epochs, quant_vector_dim):
    self.env = env

    self.action_weights = np.zeros(NO_OF_ACTIONS) 
    for weight in range(len(self.action_weights)):
      self.action_weights[weight] = 1 / NO_OF_ACTIONS

    self.QBN = QuantisedBottleneckNetwork(quant_vector_dim)
    self.training_set_size = training_set_size
    self.no_of_training_epochs = no_of_training_epochs

  def train_qbn(self):
    self.QBN, loss_values_per_epoch = self.QBN.train_network(self.env, self.training_set_size, self.no_of_training_epochs)
    # print(loss_values_per_epoch)

  # Simulation of interaction in the environment. Policy is random and doesn't change.
  # Probably worth adding code here to relate to more formalised tests
  def testing_quantisation(self, no_of_steps):
    self.QBN.eval()
    env = self.env
    board_state = env.reset()
    for _ in range(no_of_steps):
      board_state_tensor = torch.tensor(board_state)

      print("State of world before quantisation")
      print(board_state_tensor)
      print("-----------------------------------")

      quantised_encoding, decoding = self.QBN(board_state_tensor)

      print("Quantised encoding")
      print(quantised_encoding)
      print("-----------------------------------")
      print("Decoding (should be as close to original as possible)")
      print(decoding)
      print("-----------------------------------")
      print("-----------------------------------")

      action = self._choose_action()
      new_board_state, _, _, _ = env.step(action)
      board_state = new_board_state
    env.close()

  # Random policy. Subject to change if necessary
  def _choose_action(self):
    potential_actions = range(self.no_of_actions)
    return random.choices(potential_actions, weights=([0.25, 0.25, 0.25, 0.25]), k = 1)[0]