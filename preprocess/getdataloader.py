from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def Cifar10(batchsize):
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        # Discretize(), TODO: we should investigate...
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10("./data", train=True, transform=transforms_train, download=True)
    test_dataset = datasets.CIFAR10("./data", train=False, transform=transforms_test, download=True) 

    return DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True), DataLoader(test_dataset, batch_size=batchsize, drop_last=True)


def Cifar100(batchsize):
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
    ])

    train_dataset = datasets.CIFAR100("./data", train=True, transform=transforms_train, download=True)
    test_dataset = datasets.CIFAR100("./data", train=False, transform=transforms_test, download=True) 

    return DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True), DataLoader(test_dataset, batch_size=batchsize, drop_last=True)
