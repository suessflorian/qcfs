import torch
from activations import ScaledLIF, QuantisedClipFloorShiftActivation

def primed(model, t):
    def replace_activation_by_qcfs(model, t):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = replace_activation_by_qcfs(module, t)
            if isActivation(module.__class__.__name__.lower()):
                if hasattr(module, "up"):
                    model._modules[name] = QuantisedClipFloorShiftActivation(module.up.item(), t)
                else:
                    model._modules[name] = QuantisedClipFloorShiftActivation(8., t)
        return model

    def replace_maxpool2d_by_avgpool2d(model):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
            if module.__class__.__name__ == 'MaxPool2d':
                model._modules[name] = torch.nn.AvgPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                )
        return model

    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_qcfs(model, t)
    return model


def convert_snn(model, steps=16):
    def replace_activation_by_scaled_lif(model):
        for name, module in model._modules.items():
            if hasattr(module,"_modules"):
                model._modules[name] = replace_activation_by_scaled_lif(module)
            if isActivation(module.__class__.__name__.lower()):
                if hasattr(module, "up"):
                    model._modules[name] = ScaledLIF(scale=module.up.item())
                else:
                    model._modules[name] = ScaledLIF(scale=1.)
        return model

    class SpikingModel(torch.nn.Module):
        def __init__(self, model, sim_len = 16):
            super(SpikingModel, self).__init__()
            self.model = model
            self.sim_len = sim_len

        def forward(self, x):
            spikes = torch.zeros((x.size(0), 10), device=x.device)  # WARNING: assumes 10 classes, change as needed.

            for _ in range(self.sim_len):
                spikes = self.model(x) # NOTE: the overriding here.

            return spikes

    model = replace_activation_by_scaled_lif(model)
    return SpikingModel(model, steps)

def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False
