# Forked ANN-SNN Conversion via "QCFA" During Training
Codes for Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

## Usage

Train model with QCFS-Layer 

```bash
python main.py train --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l={1,..., 8}
```
Test accuracy in ann mode or snn mode

```bash
python main.py test --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t={1, ..., 8}
```
