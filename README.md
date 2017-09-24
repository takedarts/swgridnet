# SwGridNet
*Sandwiched Grid convolutional neural Network*

## What is this ?

This is an implementation of SwGridNets (*Sandwiched Grid convolutional neural Networks*) for evaluating the performances.
The experiment results are reported in a paper of SwGridNets [1].

## How to use

`src/train.py` is a bootstrap module of training.
This module trains a SwGridNet by using a dataset for image classification tasks and reports the results which contain train loss, train accuracy, test loss and test accuracy.
The results are stored into a directory which is specified as a command argument (default is `result`).

The usage is as follows:
```
usage: train.py [-h] [--N DIMENSION] [--L LENGTH] [--k CHANNELS]
                [--output OUTPUT] [--learning NAME] [--rate LEARNING_RATE]
                [--momentum MOMENTUM] [--decay WEIGHT_DECAY] [--epoch EPOCH]
                [--batchsize BATCH_SIZE] [--procsize DATA_SIZE] [--gpu GPU_ID]
                [--no-check]
                DATASET

network trainer

positional arguments:
  DATASET               dataset name (mnist, cifar10 or cifar100)

optional arguments:
  -h, --help            show this help message and exit
  --N DIMENSION, -N DIMENSION
                        dimensions of a grid layer (default: 2)
  --L LENGTH, -L LENGTH
                        side length of grid layer (default: 4)
  --k CHANNELS, -k CHANNELS
                        number of channels of a processing unit (default: 16)
  --output OUTPUT, -o OUTPUT
                        name of output directory (default: result)
  --learning NAME, -l NAME
                        name of learning rate control (default: restart)
  --rate LEARNING_RATE, -r LEARNING_RATE
                        initial leaning rate (default: 0.2)
  --momentum MOMENTUM, -m MOMENTUM
                        momentum of SGD (default: 0.9)
  --decay WEIGHT_DECAY, -d WEIGHT_DECAY
                        weight decay (default: 0.0001)
  --epoch EPOCH, -e EPOCH
                        number of epochs for training (default: 630)
  --batchsize BATCH_SIZE, -b BATCH_SIZE
                        batch size of training (default: 128)
  --procsize DATA_SIZE, -p DATA_SIZE
                        number of images at a training process (default: 128)
  --gpu GPU_ID, -g GPU_ID
                        GPU ID
  --no-check            without type check of variables
```
You can see this with command `python src/train.py --help`.

For example, when you want to confirm a performance of a SwGwidNet (N=4,L=2,k=16) with a CIFAR-10 dataset on GPU, you should execute a following command:
```
python src/train.py -N 4 -L 2 -k 16 --gpu 0 cifar10
```
The result will be stored in a directory `result`.

## Suspend and Resume
This code saves the result in a file `snapshot.npz` after each epoch. Hence, you can restart the training if you specify the same directory as an output directory.

## Results
This code outputs following files as the results:
- `snapshot.npz`: a result of the last training.
- `bestshot.npz`: network parameters which obtains the best result.
- `loss.png`: a graph of train error and test error.
- `accuracy.png`: a graph of train accuracy and test accuracy.
- `variable.dot`: a network tree with variable.
- `function.dot`: a network tree without variable.
- `out.txt`: training results (text file).
- `log.txt`: training results (json file).
- `size.txt`: a list of the parameter sizes.

You can create a image file of the network tree from `.dot` files by using Graphviz.
The command is as follows:
```
dot -Tpng function.dot -o function.png
```

## References
[1] Takeda, Atsushi. "SwGridNet: A Deep Convolutional Neural Network based on Grid Topology for Image Classification." arXiv preprint (2017) (in press).
