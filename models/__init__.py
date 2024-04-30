import torchvision.models as models
from .ResNet import *
from .VGG import *

def modelpool(arch, dataset):
    if 'cifar100' in dataset.lower():
        num_classes = 100
    else:
        num_classes = 10

    if arch.lower() == 'vgg16':
        return models.vgg16(num_classes=num_classes)
    if arch.lower() == 'vgg16-copy':
        return vgg16(num_classes=num_classes)
    elif arch.lower() == 'resnet18':
        return models.resnet18(num_classes=num_classes)
    elif arch.lower() == 'resnet34':
        return models.resnet34(num_classes=num_classes)
    elif arch.lower() == 'resnet101':
        return models.resnet101(num_classes=num_classes)
    elif arch.lower() == 'resnet18-copy' and dataset=='fashion':
        print("using resnet18single for fashion dataset")
        return resnet18single(num_classes=num_classes)
    elif arch.lower() == 'resnet18-copy': # TODO: keeping this around until we validate the torchvision models are same
        return resnet18(num_classes=num_classes)
    else:
        print("still not support this model")
        exit(0)
