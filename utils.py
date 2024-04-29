def reset_net(model):
    for _, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'ScaledLIF' in module.__class__.__name__:
            module.reset()
    return model

def regular_set(model, paras=([],[],[])):
    for _, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for _, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for _, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for _, para in module.named_parameters():
                paras[1].append(para)
    return paras

def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False
