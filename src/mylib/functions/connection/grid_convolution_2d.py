import itertools
import numpy

from chainer import configuration
from chainer import function
from chainer import cuda
from chainer.utils import type_check
from mylib.functions.activation.relu import ReLU
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.normalization.batch_normalization import BatchNormalizationFunction

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class _Function(object):
  def __init__(self, parameter_indexes):
    self.parameter_indexes = parameter_indexes
    self.inputs = None

  def forward(self, inputs):
    self.inputs = inputs[:1] + inputs[self.parameter_indexes]

    return self.run_forward(self.inputs)

  def backward(self, grads):
    return self.run_backward(self.inputs, grads)

  def run_forward(self, inputs) :
    raise NotImplementedError()

  def run_backward(self, inputs, grads):
    raise NotImplementedError()


class _Convolution2DFunction(_Function):
  def __init__(self, parameter_indexes, stride, pad):
    super().__init__(parameter_indexes)
    self.func = Convolution2DFunction(stride=stride, pad=pad)

  def run_forward(self, inputs):
    return self.func.forward(inputs)

  def run_backward(self, inputs, grads):
    return self.func.backward(inputs, grads)


class _BatchNormalizationFunction(_Function):
  def __init__(self, parameter_indexes, mean, var):
    super().__init__(parameter_indexes)
    self.func = BatchNormalizationFunction(mean=mean, var=var)

  def run_forward(self, inputs):
    if configuration.config.train:
      return self.func.forward(inputs)
    else:
      return self.func.forward(inputs + (self.func.running_mean, self.func.running_var))

  def run_backward(self, inputs, grads):
    if configuration.config.train:
      return self.func.backward(inputs, grads)
    else:
      return self.func.backward(inputs + (self.func.running_mean, self.func.running_var), grads)


class _ReLUFunction(_Function):
  def __init__(self):
    super().__init__(slice(0, 0))
    self.func = ReLU()

  def run_forward(self, inputs):
    return self.func.forward(inputs)

  def run_backward(self, inputs, grads):
    return self.func.backward(inputs, grads)


class GridConvolution2DUnitFunction(function.Function):
  def __init__(self, stride, pad, norm_stats):
    self.functions = [_BatchNormalizationFunction(slice(5, 7), *norm_stats[0:2]),
                      _Convolution2DFunction(slice(1, 3), 1, 0),
                      _BatchNormalizationFunction(slice(7, 9), *norm_stats[2:4]),
                      _ReLUFunction(),
                      _Convolution2DFunction(slice(3, 5), stride, pad),
                      _BatchNormalizationFunction(slice(9, 11), *norm_stats[4:6])]
    self.norm_stats = norm_stats

  def check_type_forward(self, in_types):
    type_check.expect(in_types.size() == 11)

    x_type = in_types[0]
    channels = [x_type.shape[1]]

    type_check.expect(x_type.dtype.kind == 'f',
                      x_type.ndim == 4)

    for i, j in enumerate(range(1, 5, 2)):
      w_type, b_type = in_types[j:j + 2]

      type_check.expect(w_type.dtype == x_type.dtype,
                        b_type.dtype == x_type.dtype,
                        w_type.ndim == 4,
                        b_type.ndim == 1,
                        w_type.shape[1] == channels[i],
                        b_type.shape[0] == w_type.shape[0])

      channels.append(w_type.shape[0])

    for i, j in enumerate(range(5, 11, 2)):
      gamma_type, beta_type = in_types[j:j + 2]

      type_check.expect(gamma_type.dtype == x_type.dtype,
                        beta_type.dtype == x_type.dtype,
                        gamma_type.ndim == 1,
                        beta_type.ndim == 1,
                        gamma_type.shape[0] == channels[i],
                        beta_type.shape[0] == channels[i])

  def forward(self, inputs):
    self.retain_inputs(())
    x = inputs[:1]
    p = inputs[1:]

    for func in self.functions:
      x = func.forward(x + p)

    return x

  def backward(self, inputs, grads):
    input_grads = [None] * len(inputs)

    for func in self.functions[::-1]:
      grads = func.backward(grads)
      input_grads[func.parameter_indexes] = grads[1:]
      grads = grads[:1]

    input_grads[0] = grads[0]

    self.norm_stats[0][:] = self.functions[0].func.running_mean
    self.norm_stats[1][:] = self.functions[0].func.running_var
    self.norm_stats[2][:] = self.functions[2].func.running_mean
    self.norm_stats[3][:] = self.functions[2].func.running_var
    self.norm_stats[4][:] = self.functions[5].func.running_mean
    self.norm_stats[5][:] = self.functions[5].func.running_var

    return tuple(input_grads)


class GridConvolution2DFunction(function.Function):
  def __init__(self, dimensions, length, stride, pad, norm_stats, noise, shake):
    self.dimensions = dimensions
    self.length = length
    self.noise = noise
    self.shake = shake
    self.functions = []
    self.connections = []
    self.shape = None

    shifts = [length ** i for i in range(dimensions)][::-1]

    for p in itertools.product(range(length), repeat=dimensions):
      idx = sum([p[d] * shifts[d] for d in range(dimensions)])
      stats = norm_stats[idx * 6:(idx + 1) * 6]

      self.functions.append(GridConvolution2DUnitFunction(stride, pad, stats))
      self.connections.append([idx - shifts[d] for d in range(dimensions) if p[d] > 0])

  def check_type_forward(self, in_types):
    type_check.expect(in_types.size() == (len(self.functions) * 10) + 1)

    x_type = in_types[0]
    in_channels = 0

    type_check.expect(x_type.dtype.kind == 'f',
                      x_type.ndim == 4)

    for i in range(len(self.functions)):
      unit_types = in_types[1 + i * 10:1 + (i + 1) * 10]
      unit_channels = [unit_types[0].shape[1]]

      in_channels += type_check.eval(unit_types[0].shape[1])

      for j, k in enumerate(range(0, 4, 2)):
        w_type, b_type = unit_types[k:k + 2]

        type_check.expect(w_type.dtype == x_type.dtype,
                          b_type.dtype == x_type.dtype,
                          w_type.ndim == 4,
                          b_type.ndim == 1,
                          w_type.shape[1] == unit_channels[j],
                          b_type.shape[0] == w_type.shape[0])

        unit_channels.append(w_type.shape[0])

      for j, k in enumerate(range(4, 10, 2)):
        gamma_type, beta_type = unit_types[k:k + 2]

        type_check.expect(gamma_type.dtype == x_type.dtype,
                          beta_type.dtype == x_type.dtype,
                          gamma_type.ndim == 1,
                          beta_type.ndim == 1,
                          gamma_type.shape[0] == unit_channels[j],
                          beta_type.shape[0] == unit_channels[j])

    type_check.expect(x_type.shape[1] == in_channels)

  def forward(self, inputs):
    self.retain_inputs(range(1, len(inputs)))
    self.shape = inputs[0].shape[:]

    xp = cuda.get_array_module(*inputs)
    in_idxs, out_idxs = self._get_indexes(inputs)
    
    in_data = inputs[0]
    out_data = xp.zeros((in_data.shape[0], out_idxs[-1][1]) + in_data.shape[2:], dtype=in_data.dtype)

    for i, func in enumerate(self.functions):
      conns = self.connections[i]
      p = inputs[1 + i * 10:1 + (i + 1) * 10]
      x = in_data[:, slice(*in_idxs[i])]
      s = (in_data.shape[0],) + ((1,) * (len(in_data.shape) - 1))
      
      if self.noise != 0.0:
        a = xp.random.normal(1.0, self.noise, s).astype(xp.float32)
      else:
        a = xp.ones(s, dtype=xp.float32)

      if self.shake != 0.0:
        conns = numpy.random.permutation(conns)
        conns = conns[int(len(conns) * self.shake):]

      if len(conns) != 0 and len(conns) != len(self.connections):
        a *= len(self.connections[i]) / len(conns)

      for c in conns:
        x += out_data[:, slice(*out_idxs[c])] * a

      if len(conns) != 0:
        x /= len(conns) + 1

      out_data[:, slice(*out_idxs[i])] = func.forward((x,) + p)[0]

    return out_data,

  def backward(self, inputs, grads):
    xp = cuda.get_array_module(*inputs)
    in_idxs, out_idxs = self._get_indexes(inputs)

    in_grads = [xp.zeros(self.shape, dtype=xp.float32)] + [None] * (len(inputs) - 1)
    out_grad = grads[0]

    for i in range(len(self.functions) - 1, -1, -1):
      p = inputs[1 + i * 10:1 + (i + 1) * 10]
      g = out_grad[:, slice(*out_idxs[i])]

      g = self.functions[i].backward((None,) + p, (g,))
      gx = g[0]
      gp = g[1:11]

      if len(self.connections[i]) != 0:
        gx /= len(self.connections[i]) + 1

      for c in self.connections[i]:
        out_grad[:, slice(*out_idxs[c])] += gx

      in_grads[0][:, slice(*in_idxs[i])] = gx
      in_grads[1 + i * 10:1 + (i + 1) * 10] = gp

    return tuple(in_grads)

  def _get_indexes(self, inputs):
    in_indexes = [(0, inputs[1].shape[1])]
    out_indexes = [(0, inputs[1].shape[0])]

    for i in range(1, len(self.functions)):
      w = inputs[1 + i * 10]
      in_indexes.append((in_indexes[-1][1], in_indexes[-1][1] + w.shape[1]))
      out_indexes.append((out_indexes[-1][1], out_indexes[-1][1] + w.shape[0]))

    return in_indexes, out_indexes
