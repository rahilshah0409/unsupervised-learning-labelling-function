import torch
import tools as tl
import torch.nn as nn
from torch.autograd import Variable
import gym

if __name__ == '__main__':
  env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0", params={"generation": "random", "environment_seed": 0})
  testing_set_size = 1000
  batch_size = 64
  total_batches = testing_set_size / batch_size
  if testing_set_size % batch_size != 0:
      total_batches += 1

  qbn = torch.load("best_model.pt")

  testing_set = tl.generate_testing_data(env, testing_set_size)

  loss_total = 0
  with torch.no_grad():
     for b_i in range(total_batches):
      batch_input = testing_set[(b_i * batch_size) : (b_i * batch_size) + batch_size]
      batch_input = Variable(torch.FloatTensor(batch_input))
      batch_target = Variable(torch.FloatTensor(batch_input))
      if torch.cuda.is_available():
        batch_target, batch_input = batch_target.cuda(), batch_input.cuda()
      
      encoding, reconstruction = qbn(batch_input)

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

  print("Average loss: %d", loss_total / len(testing_set))
                    