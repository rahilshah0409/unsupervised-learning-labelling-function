import torch.nn as nn
import sys
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/AEExperiment/AEExperiment")
from waterWorld.qbnTraining.quantisationMethods import BinarySigmoid

class QuantisedBottleneckNetwork(nn.Module):

    def __init__(self, input_vec_dim, quant_vector_dim=6, batch_size=32, learning_rate=0.005, weight_decay=0.01, epochs=100, encoder_activation=BinarySigmoid(), training_set_size=2000):
        super(QuantisedBottleneckNetwork, self).__init__()
        self.input_vec_dim = input_vec_dim
        print("Input vec dim")
        print(self.input_vec_dim)
        self.quant_vector_dim = quant_vector_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.training_set_size = training_set_size
        f1 = 4 * self.quant_vector_dim
        f2 = 2 * self.quant_vector_dim
        self.encoder = nn.Sequential(nn.Linear(self.input_vec_dim, f1),
                                     nn.Tanh(),
                                     # nn.Linear(f1, f2),
                                     # nn.Tanh(),
                                     nn.Linear(f1, self.quant_vector_dim),
                                     encoder_activation)
        self.decoder = nn.Sequential(nn.Linear(self.quant_vector_dim, f1),
                                     nn.Tanh(),
                                     # nn.Linear(f2, f1),
                                     # nn.Tanh(),
                                     nn.Linear(f1, self.input_vec_dim))

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
