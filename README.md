# Forked ANN-SNN Conversion via "QCFA" During Training
Codes for Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

## General Usage

Train model with QCFS-Layer 

```bash
python main.py train --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l={1,..., 8}
```
Test accuracy in ann mode or snn mode

```bash
python main.py test --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t={1, ..., 8}
```
## Existing Model Parameter Dict
I managed to shoehorn a full model with name `resnet18-default` in the repo ready for your perusal (trained via `... train --bs=128 --model=resnet18 --data=cifar10 --id=resnet18-default --l=16`). Can test on `CIFAR-10` via

ANN variant;
```
pipenv run python main.py test --model resnet18 --id=resnet18-default --mode=ann
```

But more importantly, the SNN variant;
```
pipenv run python main.py test --model resnet18 --id=resnet18-default --mode=snn --t=16
```

## Other notes;
The VGG16 sits in the ~86% realm with default `l=16` and `t=16`.
