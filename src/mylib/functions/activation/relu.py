#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ReLU(function.Function):
  def check_type_forward(self, in_types):
    type_check.expect(in_types.size() == 1,
                      in_types[0].dtype.kind == 'f')

  def forward(self, inputs):
    xp = cuda.get_array_module(*inputs)
    
    return xp.maximum(inputs[0], 0.0, dtype=inputs[0].dtype),
        
  def backward(self, inputs, grads):
    return (grads[0] * (inputs[0] > 0.0)).astype(dtype=grads[0].dtype),


def relu(x):
  return ReLU()(x)
  
