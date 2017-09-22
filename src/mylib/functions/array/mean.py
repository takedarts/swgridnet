#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class MeanFunction(function.Function):
  def check_type_forward(self, in_types):
    type_check.expect(in_types.size() >= 1)

    type_check.expect(in_types[0].dtype.kind == 'f',
                      in_types[0].ndim >= 2)

    for i in range(1, type_check.eval(in_types.size())):
      type_check.expect(in_types[i].dtype == in_types[0].dtype,
                        in_types[i].ndim == in_types[0].ndim,
                        in_types[i].shape[0] == in_types[0].shape[0])

      for d in range(2, type_check.eval(in_types[0].ndim)):
        type_check.expect(in_types[i].shape[d] == in_types[0].shape[d])

  def forward(self, inputs):
    self.retain_inputs(())
    self._channels = [x.shape[1] for x in inputs]

    xp = cuda.get_array_module(*inputs)
    channels = max(self._channels)
    
    shape = [1, channels] + [1] * (inputs[0].ndim - 2)
    d = xp.zeros(shape, dtype=inputs[0].dtype)
    
    shape = [inputs[0].shape[0], channels] + list(inputs[0].shape[2:])
    y = xp.zeros(shape, dtype=inputs[0].dtype)

    for x in inputs: 
      d[:, :x.shape[1]] += 1
      y[:, :x.shape[1]] += x

    y /= d

    return y,

  def backward(self, inputs, grads):
    xp = cuda.get_array_module(*grads)
    channels = max(self._channels)
    
    shape = [1, channels] + [1] * (grads[0].ndim - 2)
    d = xp.zeros(shape, dtype=grads[0].dtype)
    
    for c in self._channels: 
      d[:, :c] += 1
      
    g = grads[0]
    g /= d
    
    gx = [g[:, :c] for c in self._channels]
    
    return tuple(gx)


def mean(*args):
  return MeanFunction()(*args)

