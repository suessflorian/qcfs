import torch
from activations import ScaledLIF, QuantisedClipFloorShiftActivation

def primed(model, t):
    def replace_activation_by_qcfs(model, t):
        count, scaledCount = 0, 0
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name], added, scaledAdded = replace_activation_by_qcfs(module, t)
                count += added
                scaledCount += scaledAdded
            if isActivation(module.__class__.__name__.lower()):
                count+=1
                if hasattr(module, "up"):
                    scaledCount+=1
                    model._modules[name] = QuantisedClipFloorShiftActivation(module.up.item(), t)
                else:
                    model._modules[name] = QuantisedClipFloorShiftActivation(8., t)
        return model, count, scaledCount

    def replace_maxpool2d_by_avgpool2d(model):
        count = 0
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name], added = replace_maxpool2d_by_avgpool2d(module)
                count += added
            if module.__class__.__name__ == 'MaxPool2d':
                count+=1
                model._modules[name] = torch.nn.AvgPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                )
        return model, count

    model, count = replace_maxpool2d_by_avgpool2d(model)
    print(f"replaced {count} pool(s) with average pool equivalents.")

    model, count, scaledCount = replace_activation_by_qcfs(model, t)
    print(f"replaced {count} activation(s) with QCFS for {t}.")
    print(f"{scaledCount} of have learnable scale parameters.")

    return model


def convert_snn(model, steps=16):
    def replace_activation_by_scaled_lif(model):
        count, scaledCount = 0, 0
        for name, module in model._modules.items():
            if hasattr(module,"_modules"):
                model._modules[name] = replace_activation_by_scaled_lif(module)
            if isActivation(module.__class__.__name__.lower()):
                count+=1
                if hasattr(module, "up"):
                    scaledCount += 1
                    model._modules[name] = ScaledLIF(scale=module.up.item())
                else:
                    model._modules[name] = ScaledLIF(scale=1.)

        print(f"replaced {count} activation(s) with scaled LIFS.")
        print(f"{scaledCount} of those came with learnt scale parameters.")
        return model

    class SpikingModel(torch.nn.Module):
        def __init__(self, model, sim_len = 16):
            super(SpikingModel, self).__init__()
            self.model = model
            self.sim_len = sim_len

        def forward(self, x):
            for _ in range(self.sim_len):
                spikes = self.model(x) # NOTE: the overriding here.

            return spikes

    model = replace_activation_by_scaled_lif(model)
    return SpikingModel(model, steps)

def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False
