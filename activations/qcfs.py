import torch

# Floor straight-through estimated.
class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantisedClipFloorShiftActivation(torch.nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = torch.nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        x = Floor.apply(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x
