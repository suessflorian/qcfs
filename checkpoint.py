import os
import torch

def load(model, id):
    checkpoint_path = f'./saved_models/{id}.pth'
    if os.path.isfile(checkpoint_path):
        print('checkpoint found... retrieving')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        return model

    print("no checkpoint found... assume novel")
    return model

def save(model, id):
    torch.save(model.state_dict(), f'./saved_models/{id}.pth')
