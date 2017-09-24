#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sandwiched Grid convolutional neural Network Module.
Unit: BN - Conv1x1 - BN - ReLU - Conv3x3 - BN
Split: Conv1x1 - BN
Join: ReLU - Conv1x1 - BN
Process: add - call - append
@author: Atsushi TAKEDA
'''

CHANNELS = 64

import chainer
import mylib

class Block(chainer.Chain):
  def __init__(self, in_channels, out_channels, dimensions, length, filters):
    super().__init__()

    grid = mylib.links.GridConvolution2D(dimensions, length, filters)
    grid_input = grid.get_input_channels()
    grid_output = grid.get_output_channels()

    self.add_link('norm0', chainer.links.BatchNormalization(in_channels))
    self.add_link('conv1', chainer.links.Convolution2D(in_channels, grid_input, 1))
    self.add_link('norm1', chainer.links.BatchNormalization(grid_input))
    self.add_link('grid', grid)
    self.add_link('conv2', chainer.links.Convolution2D(grid_output, out_channels, 1))
    self.add_link('norm2', chainer.links.BatchNormalization(out_channels))

  def __call__(self, x):
    y = x

    x = self.norm0(x)
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.grid(x)
    x = chainer.functions.relu(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = mylib.functions.add(x, y)

    return x


class Network(chainer.Chain):
  def __init__(self, category, dimension, length, filters):
    super().__init__(input=chainer.links.Convolution2D(None, CHANNELS * 1, 3, pad=1),
                     norm=chainer.links.BatchNormalization(CHANNELS * 1),
                     block1=Block(CHANNELS * 1, CHANNELS * 2, dimension, length, (filters * 1, filters * 2)),
                     block2=Block(CHANNELS * 2, CHANNELS * 4, dimension, length, (filters * 2, filters * 4)),
                     block3=Block(CHANNELS * 4, CHANNELS * 8, dimension, length, (filters * 4, filters * 8)),
                     output=chainer.links.Linear(CHANNELS * 8, category))

  def __call__(self, x):
    x = self.input(x)
    x = self.norm(x)

    x = self.block1(x)
    x = chainer.functions.average_pooling_2d(x, 2)

    x = self.block2(x)
    x = chainer.functions.average_pooling_2d(x, 2)

    x = self.block3(x)

    x = chainer.functions.relu(x)
    x = chainer.functions.average_pooling_2d(x, x.shape[2])
    x = self.output(x)

    return x

