from .getdataloader import *

def datapool(DATANAME, batchsize):
    if DATANAME.lower() == 'cifar10':
        return Cifar10(batchsize)
    elif DATANAME.lower() == 'fashion':
        return FashionMNIST(batchsize)
    elif DATANAME.lower() == 'cifar100':
        return Cifar100(batchsize)
    else:
        raise ValueError(f"dataset {DATANAME} not yet supported...")
