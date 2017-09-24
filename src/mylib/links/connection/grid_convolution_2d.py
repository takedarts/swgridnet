import numpy

from chainer import initializers
from chainer import link
from mylib.functions.connection.grid_convolution_2d import GridConvolution2DUnitFunction
from mylib.functions.connection.grid_convolution_2d import GridConvolution2DFunction

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class GridConvolution2DUnit(link.Link):
  def __init__(self, in_filters, out_filters, ksize=3, stride=1, pad=1, decay=0.9, eps=2e-5,
               initial_weight=None, initial_bias=None, initial_gamma=None, initial_beta=None):
    super().__init__()

    self.stride = _pair(stride)
    self.pad = _pair(pad)

    self._initialize_weight('W1', in_filters, out_filters, 1, initial_weight)
    self._initialize_weight('W2', out_filters, out_filters, ksize, initial_weight)

    self._initialize_bias('b1', out_filters, initial_bias)
    self._initialize_bias('b2', out_filters, initial_bias)

    self._initialize_gamma('gamma0', in_filters, initial_gamma)
    self._initialize_gamma('gamma1', out_filters, initial_gamma)
    self._initialize_gamma('gamma2', out_filters, initial_gamma)

    self._initialize_beta('beta0', in_filters, initial_beta)
    self._initialize_beta('beta1', out_filters, initial_beta)
    self._initialize_beta('beta2', out_filters, initial_beta)

    self.add_persistent('mean0', numpy.zeros(in_filters, dtype=numpy.float32))
    self.add_persistent('mean1', numpy.zeros(out_filters, dtype=numpy.float32))
    self.add_persistent('mean2', numpy.zeros(out_filters, dtype=numpy.float32))

    self.add_persistent('var0', numpy.zeros(in_filters, dtype=numpy.float32))
    self.add_persistent('var1', numpy.zeros(out_filters, dtype=numpy.float32))
    self.add_persistent('var2', numpy.zeros(out_filters, dtype=numpy.float32))

  def _initialize_weight(self, name, in_filters, out_filters, ksize, initializer):
    if initializer is None:
      initializer = initializers.HeNormal(dtype=numpy.float32)

    self.add_param(name, (out_filters, in_filters) + tuple(_pair(ksize)[:2]), initializer=initializer)

  def _initialize_bias(self, name, filters, initializer):
    if initializer is None:
      initializer = initializers.Zero(dtype=numpy.float32)

    self.add_param(name, (filters,) , initializer=initializer)

  def _initialize_gamma(self, name, filters, initializer):
    if initializer is None:
      initializer = initializers.One(dtype=numpy.float32)

    self.add_param(name, (filters,) , initializer=initializer)

  def _initialize_beta(self, name, filters, initializer):
    if initializer is None:
      initializer = initializers.Zero(dtype=numpy.float32)

    self.add_param(name, (filters,) , initializer=initializer)

  def __call__(self, x):
    stats = (self.mean0, self.var0, self.mean1, self.var1, self.mean2, self.var2)
    params = (self.W1, self.b1, self.W2, self.b2,
              self.gamma0, self.beta0, self.gamma1, self.beta1, self.gamma2, self.beta2)
    func = GridConvolution2DUnitFunction(self.stride, self.pad, stats)

    return func(x, *params)


class GridConvolution2D(link.ChainList):
  def __init__(self, dimensions, length, filters, ksize=3, stride=1, pad=1, decay=0.9, eps=2e-5,
               initial_weight=None, initial_bias=None, initial_gamma=None, initial_beta=None):
    super().__init__()

    self.dimensions = dimensions
    self.length = length
    self.stride = stride
    self.pad = pad

    self.in_channels = 0
    self.out_channels = 0

    filters = _pair(filters)

    if len(filters) == 2:
      a, b = filters
      s = 1 + length + ((length - 1) * (dimensions - 1))
      filters = [int(a * (s - 1 - i) / (s - 1) + b * i / (s - 1)) for i in range(s)]

    shifts = [length ** i for i in range(dimensions)]

    for i in range(length ** dimensions):
      s = sum([int(i / s) % length for s in shifts])
      self.in_channels += filters[s]
      self.out_channels += filters[s + 1]
      self.add_link(GridConvolution2DUnit(filters[s], filters[s + 1], ksize, stride, pad, decay, eps,
                                          initial_weight, initial_bias, initial_gamma, initial_beta))

  def get_input_channels(self):
    return self.in_channels

  def get_output_channels(self):
    return self.out_channels

  def __call__(self, x):
    stats = []
    args = []

    for u in self:
      stats.extend([u.mean0, u.var0, u.mean1, u.var1, u.mean2, u.var2])
      args.extend([u.W1, u.b1, u.W2, u.b2, u.gamma0, u.beta0, u.gamma1, u.beta1, u.gamma2, u.beta2])

    func = GridConvolution2DFunction(self.dimensions, self.length, self.stride, self.pad, stats)

    return func(x, *args)

