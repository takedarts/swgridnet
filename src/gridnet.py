#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Grid Network モジュール。
Unit: BN - Conv1x1 - BN - ReLU - Conv3x3 - BN
Split: Conv1x1  - BN
Join: ReLU - Conv1x1 - BN
Process: add - call - append
@author: Atsushi TAKEDA
'''

DEPTH = 1
DIMENSIONS = 2
LENGTH = 5
CHANNELS = 64
FILTERS = 32
DROP = 0.0
NOISE = 0.0
SHAKE = 0.0

import chainer
import mylib

class Block(chainer.Chain):
  def __init__(self, in_channels, out_channels, dimensions, length, filters, drop, noise, shake):
    super().__init__()

    grid = mylib.links.GridConvolution2D(dimensions, length, filters, noise=noise, shake=shake)
    grid_input = grid.get_input_channels()
    grid_output = grid.get_output_channels()

    self.add_link('norm0', chainer.links.BatchNormalization(in_channels))
    self.add_link('conv1', chainer.links.Convolution2D(in_channels, grid_input, 1))
    self.add_link('norm1', chainer.links.BatchNormalization(grid_input))
    self.add_link('grid', grid)
    self.add_link('conv2', chainer.links.Convolution2D(grid_output, out_channels, 1))
    self.add_link('norm2', chainer.links.BatchNormalization(out_channels))

    self.drop = drop

  def __call__(self, x):
    y = x

    x = self.norm0(x)
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.grid(x)
    x = chainer.functions.dropout(x, self.drop)
    x = chainer.functions.relu(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = mylib.functions.add(x, y)

    return x


class Section(chainer.ChainList):
  def __init__(self, in_channels, out_channels, depth, dimensions, length, filters, drop, noise, shake):
    super().__init__()

    self.add_link(Block(in_channels, out_channels, dimensions, length, filters, drop, noise, shake))

    for _ in range(depth - 1):
      self.add_link(Block(out_channels, out_channels, dimensions, length, filters, drop, noise, shake))

  def __call__(self, x):
    for block in self:
      x = block(x)

    return x


class Network(chainer.Chain):
  def __init__(self, category):
    depth = DEPTH
    dimensions = DIMENSIONS
    length = LENGTH
    channels = CHANNELS
    filters = FILTERS
    drop = DROP / 10
    noise = NOISE / 10
    shake = SHAKE / 10

    super().__init__(input=chainer.links.Convolution2D(None, channels * 1, 3, pad=1),
                     norm=chainer.links.BatchNormalization(channels * 1),
                     sec1=Section(channels * 1, channels * 2, depth, dimensions, length,
                                  (filters * 1, filters * 2), drop, noise, shake),
                     sec2=Section(channels * 2, channels * 4, depth, dimensions, length,
                                  (filters * 2, filters * 4), drop, noise, shake),
                     sec3=Section(channels * 4, channels * 8, depth, dimensions, length,
                                  (filters * 4, filters * 8), drop, noise, shake),
                     output=chainer.links.Linear(channels * 8, category))

  def __call__(self, x):
    x = self.input(x)
    x = self.norm(x)

    x = self.sec1(x)
    x = chainer.functions.average_pooling_2d(x, 2)

    x = self.sec2(x)
    x = chainer.functions.average_pooling_2d(x, 2)

    x = self.sec3(x)

    x = chainer.functions.relu(x)
    x = chainer.functions.average_pooling_2d(x, x.shape[2])
    x = self.output(x)

    return x

