import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from quantisationMethods import BinarySigmoid
import tools
import torch.optim as optim
from numpy.linalg import norm

class QuantisedBottleneckNetwork(nn.Module):

  # Understand what the difference is between input size and x_features in the code
  # Hard coded the values based on my understanding of the code and the input vectors we expect 
  # Using binary quantisation
  # Introduced hyperparameter of the dimension of the quantised vector, not sure if I want to introduce it here
  def __init__(self, quant_vector_dim=6, batch_size=32, learning_rate=0.005, weight_decay=0.01, epochs=100, training_set_size=2000):
    super(QuantisedBottleneckNetwork, self).__init__()
    self.quant_vector_dim = quant_vector_dim
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.epochs = epochs
    self.training_set_size = training_set_size
    self.encoder = nn.Sequential(nn.Linear(52, 416),
                                  nn.Tanh(),
                                  nn.Linear(416, self.quant_vector_dim),
                                  BinarySigmoid())
    self.decoder = nn.Sequential(nn.Linear(self.quant_vector_dim, 416),
                                  nn.Tanh(),
                                  nn.Linear(416, 52),
                                  nn.ReLU6())

  # Method that mimics a forward pass in the QBN
  def forward(self, input):
    encoding = self.encode(input)
    decoding = self.decode(encoding)

    return encoding, decoding

  # Method that calls the encoder to encode the input vector x
  def encode(self, x):
    return self.encoder(x)

  # Method that calls the decoder to decode the input vector x
  def decode(self, x):
    return self.decoder(x)
  
  def fit(self, x_train, y_train):
    return self.fit(self, x_train)
  
  def fit(self, obs_training_data):
    # Use MSE loss and Adam optimiser with learning rate and weight decay hyperparameters
    mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    quantised_vectors = []
    loss_values_per_epoch = []
    total_batches = math.ceil(self.training_set_size / self.batch_size)

    # QBN training loop
    print("Beginning training of QBN")
    for epoch in range(self.epochs):
        self.train()
        loss_values = []
        random.shuffle(obs_training_data)
        for b_i in range(total_batches):
            batch_input = obs_training_data[(b_i * self.batch_size) : (b_i * self.batch_size) + self.batch_size]
            batch_input = torch.FloatTensor(batch_input)
            batch_target = Variable(batch_input)
            batch_input = Variable(batch_input, requires_grad=True)

            if (torch.cuda.is_available()):
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            quantised_vector, feature_reconstruction = self.forward(batch_input)
            quantised_vectors.append(quantised_vector)

            optimizer.zero_grad()
            loss = mse_loss(feature_reconstruction, batch_target)
            loss.backward()
            # Loss value rounded to 2dp before adding to record of loss values
            # loss_values.append(round(loss.item(), 2))
            optimizer.step()

            print('Epoch: %d, Loss: %f' % (epoch, loss.item()))

  def score(self, x, y):
    y_pred = self(x)
    return np.dot(y_pred, y)/(norm(y_pred)*norm(y))
  
  def get_params(self, deep=False):
    return {"quant_vector_dim": self.quant_vector_dim, "batch_size": self.batch_size, "learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "epochs": self.epochs}

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
        setattr(self, parameter, value)
    return self
  
  # Method that triggers the training of the network
  # Ignoring details about cuda for now
  # They perform training in batches, I might ignore this for now (don't see the necessity for our case)
  def train_network(self, env, dataset_size, epochs):
    self.train()

    obs_training_data = tools.generate_training_data(env, dataset_size)
    # Why MSE loss?
    mse_loss = nn.MSELoss()
    # Don't know how to use Adam optimiser and why we are using it. LOOK INTO THIS MORE 
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)
    quantised_vectors = []
    loss_values_per_epoch = []

    # Training loop, perform forward passes and back propogation for each epoch
    for epoch in range(epochs):
      print("Starting epoch number {}".format(epoch))
      loss_values = []
      self.train()
      random.shuffle(obs_training_data)
      for _, input in enumerate(obs_training_data):
        # Not sure why Variables are needed to wrap the Tensor objects or why requires_grad needs to be true
        tensor_input = torch.FloatTensor(input)
        target = Variable(tensor_input)
        input = Variable(tensor_input, requires_grad=True)
        decoded_output, encoded_input = self(input)
        quantised_vectors.append(encoded_input)

        optimizer.zero_grad()
        loss = mse_loss(decoded_output, target)
        loss.backward()
        loss_values.append(round(loss.item(), 2))  
        optimizer.step()

      loss_values_per_epoch.append(loss_values)

    # TODO: plot the progress of the loss value as indicator of the model training
    return self, loss_values_per_epoch
