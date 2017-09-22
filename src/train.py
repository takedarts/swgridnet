#!/usr/bin/env python
# -*- coding: utf-8 -*-

import swgridnet
import mylib

import os
import argparse
import chainer
from chainer.training import extensions


def load_dataset(name):
  '''load the specified datasets'''
  if name == 'mnist':
    return (10,) + mylib.datasets.get_mnist()
  elif name == 'cifar10':
    return (10,) + mylib.datasets.get_cifar10()
  elif name == 'cifar100':
    return (100,) + mylib.datasets.get_cifar100()
  
  
def create_network(cls, category):
  '''create the specified network model'''
  network = cls(category)
  lossfun = chainer.functions.softmax_cross_entropy
  accfun = chainer.functions.accuracy

  return chainer.links.Classifier(network, lossfun=lossfun, accfun=accfun)


def _make_argument_values(args):
  '''make values from the specified arguments'''
  values = (('Dataset', args.dataset),
            ('LearningType', args.learning),
            ('LearningRate', args.rate),
            ('Momentum', args.momentum),
            ('WeightDecay', args.decay),
            ('Epoch', args.epoch),
            ('BatchSize', args.batchsize))

  return tuple([(k, str(v).strip()) for k, v in values])


def save_arguments(path, args):
  '''save argument values to the specified file'''
  file = os.path.join(path, 'argument.txt')
  values = _make_argument_values(args)
  
  with open(file, 'w') as handle:
    for k, v in values:
      handle.write("{}: {}\n".format(k, v))


def load_arguments(path):
  '''load argument values from the specified file'''
  file = os.path.join(path, 'argument.txt')

  if not os.path.isfile(file):
    return None

  with open(file, 'r') as handle:
    vals = tuple([tuple(line.strip().split(': ')) for line in handle])

  return vals


def check_arguments(path, args):
  '''check argument values from the specified file'''
  vals1 = load_arguments(path)
  
  if vals1 is None:
    return True
  
  vals1 = dict(vals1)
  vals2 = dict(_make_argument_values(args))
  
  if vals1['LearningType'] == 'restart' and vals2['LearningType'] == 'restart':
    del vals1['Epoch']
    del vals2['Epoch']
  
  return vals1 == vals2


def main():
  parser = argparse.ArgumentParser(description='network trainer')
  parser.add_argument('dataset', metavar='DATASET', help='dataset name (mnist, cifar10 or cifar100)')
  parser.add_argument('--output', '-o', default='result', metavar='OUTPUT',
                      help='name of output directory (default: result)')
  parser.add_argument('--learning', '-l', default='restart', choices=('step', 'cosine', 'restart'),
                      metavar='NAME', help='name of learning rate control (default: restart)')
  parser.add_argument('--rate', '-r', type=float, default=0.2,
                      metavar='LEARNING_RATE', help='initial leaning rate (default: 0.2)')
  parser.add_argument('--momentum', '-m', type=float, default=0.9,
                      metavar='MOMENTUM', help='momentum of SGD (default: 0.9)')
  parser.add_argument('--decay', '-d', type=float, default=0.0001,
                      metavar='WEIGHT_DECAY', help='weight decay (default: 0.0001)')
  parser.add_argument('--epoch', '-e', type=int, default=630,
                      metavar='EPOCH', help='number of epochs for training (default: 630)')
  parser.add_argument('--batchsize', '-b', type=int, default=128,
                      metavar='BATCH_SIZE', help='batch size of training (default: 128)')
  parser.add_argument('--procsize', '-p', type=int, default=None,
                      metavar='DATA_SIZE', help='number of images at a training process (default: 128)')
  parser.add_argument('--gpu', '-g', type=int, default=-1,
                      metavar='GPU_ID', help='GPU ID')
  parser.add_argument('--no-check', action='store_true', default=False, help='without type check of variables')
  args = parser.parse_args()

  if args.procsize is None:
    args.procsize = args.batchsize

  if args.no_check:
    chainer.config.type_check = False

  # output path
  output_dir = args.output

  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  # check arguments
  if not check_arguments(output_dir, args):
    raise RuntimeError("Another settings exist in '{}'".format(output_dir))

  # load data-set
  category, train_data, test_data = load_dataset(args.dataset)

  # create a neural network
  network = create_network(swgridnet.Network, category)
  
  if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    network.to_gpu()

  # create optimizer
  optimizer = chainer.optimizers.MomentumSGD(lr=args.rate, momentum=args.momentum)
  optimizer.setup(network)
  optimizer.add_hook(chainer.optimizer.WeightDecay(args.decay))

  # create data iterators
  train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize, repeat=True, shuffle=True)
  test_iter = chainer.iterators.SerialIterator(test_data, args.procsize, repeat=False, shuffle=False)

  # create trainer
  updater = mylib.training.StandardUpdater(train_iter, optimizer, device=args.gpu, procsize=args.procsize)
  trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)

  # extension for evaluation
  trainer.extend(extensions.Evaluator(test_iter, network, device=args.gpu))

  # extension for controlling learning rate
  if args.learning == 'step':
    trainer.extend(mylib.training.extensions.StepShift('lr', args.rate, args.epoch))
  elif args.learning == 'cosine':
    trainer.extend(mylib.training.extensions.CosineShift('lr', args.rate, args.epoch, 1))
  elif args.learning == 'restart':
    trainer.extend(mylib.training.extensions.CosineShift('lr', args.rate, 10, 2))

  # extensions for logging
  plot_err_keys = ['main/loss', 'validation/main/loss']
  plot_acc_keys = ['main/accuracy', 'validation/main/accuracy']
  print_keys = ['epoch',
                'main/loss', 'validation/main/loss',
                'main/accuracy', 'validation/main/accuracy',
                'elapsed_time']
  trigger = mylib.training.trigger.IntervalTrigger

  trainer.extend(mylib.training.extensions.dump_graph('main/loss', out_name="variable.dot", remove_variable=False))
  trainer.extend(mylib.training.extensions.dump_graph('main/loss', out_name="function.dot", remove_variable=True))
  trainer.extend(mylib.training.extensions.dump_network_size(filename='size.txt'))

  trainer.extend(extensions.snapshot(filename='snapshot.npz'), trigger=trigger(1, 'epoch'))
  trainer.extend(mylib.training.extensions.Bestshot(filename='bestshot.npz', trigger=trigger(1, 'epoch')))

  trainer.extend(mylib.training.extensions.LogReport(log_name='log.txt', trigger=trigger(1, 'epoch')))
  trainer.extend(mylib.training.extensions.PrintReport(print_keys, log_report='LogReport'))
  trainer.extend(mylib.training.extensions.PrintReport(print_keys, log_report='LogReport', out='out.txt'))

  trainer.extend(mylib.training.extensions.PlotReport(plot_err_keys, 'epoch', file_name='loss.png',
                                                      marker=None, trigger=trigger(1, 'epoch')))
  trainer.extend(mylib.training.extensions.PlotReport(plot_acc_keys, 'epoch', file_name='accuracy.png',
                                                      marker=None, trigger=trigger(1, 'epoch')))

  # save arguments
  save_arguments(output_dir, args)

  # resume setting
  snapshot = os.path.join(output_dir, 'snapshot.npz')

  if os.path.isfile(snapshot):
    chainer.serializers.load_npz(snapshot, trainer)

  # start
  trainer.run()


if __name__ == '__main__':
  main()

