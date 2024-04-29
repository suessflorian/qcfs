import argparse
from models import modelpool
from preprocess import datapool
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
from checkpoint import load
import torch.nn as nn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('action', default='train', type=str, help='train or test')
    parser.add_argument('--bs', default=128, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate') 
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=120, type=int, help='Training epochs') # better if set to 300 for CIFAR dataset
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--device', default='mps', type=str, help='mps or cpu')
    parser.add_argument('--l', default=16, type=int, help='L')
    parser.add_argument('--t', default=16, type=int, help='T')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='vgg16')
    args = parser.parse_args()
    
    seed_all()

    # preparing data
    train, test = datapool(args.data, args.bs)
    # preparing model
    model = modelpool(args.model, args.data)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=args.l)
    model = load(model, args.id)

    # conversion
    if args.mode == 'snn':
        model = replace_activation_by_neuron(model)

    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    if args.action == 'train':
        train_ann(train, test, model, args.epochs, args.device, criterion, args.lr, args.wd, args.id)
    elif args.action == 'test':
        if args.mode == 'snn':
            acc = eval_snn(test, model, args.device, args.t)
            print('Accuracy: ', acc)
        elif args.mode == 'ann':
            acc, _ = eval_ann(test, model, criterion, args.device)
            print('Accuracy: {:.4f}'.format(acc))
        else:
            raise ValueError('Unrecognized mode')
