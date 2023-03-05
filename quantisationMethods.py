import torch
import torch.nn as nn
from torch.autograd import Function

class TernaryTanh(nn.Module):
  """
  reference: https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
  """

  def __init__(self):
      super(TernaryTanh, self).__init__()

  def forward(self, input):
      output = 1.5 * torch.tanh(input) + 0.5 * torch.tanh(-3 * input)
      output = ternarizeTanh(output)
      return output

class TernaryTanhF(Function):

  @staticmethod
  def forward(cxt, input):
      output = input.new(input.size())
      output.data = input.data
      output.round_()
      return output

  @staticmethod
  def backward(cxt, grad_output):
      grad_input = grad_output.clone()
      return grad_input

# ------------------------------------------------------------------------------------------

class BinaryTanh(nn.Module):
    """
    reference: https://github.com/DingKe/pytorch_workplace/blob/master/binary/modules.py#L10
    """

    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Tanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarizeTanh(output)
        return output

class BinaryTanhF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
# ------------------------------------------------------------------------------------------

class BinarySigmoid(nn.Module):
    def __init__(self):
        super(BinarySigmoid, self).__init__()
        self.hardsigmoid = nn.Sigmoid()

    def forward(self, input, stochastic=False):
        output = self.hardsigmoid(input)
        if not stochastic:
            output = binarizeSig(output)
        else:
            output = bernolliSample(output)
        return output
    
class BinarizeSigF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0.5] = 1
        output[input < 0.5] = 0
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class BernolliSampleBinarizeF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output = torch.bernoulli(output)
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

binarizeSig = BinarizeSigF.apply
bernolliSample = BernolliSampleBinarizeF.apply
binarizeTanh = BinaryTanhF.apply
ternarizeTanh = TernaryTanhF.apply