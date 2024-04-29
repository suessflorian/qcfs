from torch import nn
import torch
import snntorch as snn
from torch.autograd import Function

class ScaledNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = snn.Leaky(beta=1, reset_delay=False)


    def forward(self, x):          
        x = x / self.scale
        if self.t == 0:
            _, self.mem = self.neuron(torch.ones_like(x)*0.5, self.mem)
        self.t += 1
        x, self.mem = self.neuron(x, self.mem)

        return x * self.scale # NOTE: will this cause output to become continous?


    def reset(self): # WARNING: must call prior to using this neuron for tensor device coordination
        self.t = 0
        self.mem = self.neuron.reset_mem()

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class MyFloor(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x
