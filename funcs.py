import numpy as np
import torch
from tqdm import tqdm
from utils import *
import random
import os
import checkpoint
from tqdm import tqdm

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def eval_ann(test_dataloader, model, loss_fn, device):
    epoch_loss = 0
    tot = torch.tensor(0.).to(device)
    model.eval()
    model.to(device)
    length = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, unit="batch", desc="Evaluating"):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length

def eval_snn(test_dataloader, model, device):
    quantity_evaluated = 0
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, unit="batch"):
            reset_net(model)
            img, label = img.to(device), label.to(device)

            spikes = model(img)
            _, predicted = spikes.max(1)
            total_correct += (predicted == label).sum().item()
            quantity_evaluated += len(label)
    return total_correct / quantity_evaluated

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, id=None):
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD([
        {'params': para1, 'weight_decay': wd}, 
        {'params': para2, 'weight_decay': wd}, 
        {'params': para3, 'weight_decay': wd}
    ], lr=lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        length = 0
        model.train()
        for img, label in tqdm(train_dataloader, desc='Epoch {}'.format(epoch, unit="batch"), unit="batch"):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
            length += len(label)

        if epoch % 5 == 0:
            tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device)
            print(f'checkpoint loss: {val_loss}, test accuracy: {tmp_acc}', flush=True)
            if id != None and tmp_acc >= best_acc:
                print(f'improved, checkpoint save...')
                checkpoint.save(model, id)
            best_acc = max(tmp_acc, best_acc)
        scheduler.step()
    return best_acc, model
