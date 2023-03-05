import torch.nn as nn
from quantisationMethods import BinarySigmoid

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
