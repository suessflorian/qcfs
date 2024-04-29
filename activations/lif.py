import torch
import snntorch

class ScaledLIF(torch.nn.Module):
    def __init__(self, scale=1.):
        super(ScaledLIF, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = snntorch.Leaky(beta=1, reset_delay=False)

    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            _, self.mem = self.neuron(torch.ones_like(x)*0.5, self.mem)
        self.t += 1
        x, self.mem = self.neuron(x, self.mem)
        return x * self.scale


    def reset(self):
        self.t = 0
        self.mem = self.neuron.reset_mem()
