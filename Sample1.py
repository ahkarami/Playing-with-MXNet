# Amir Hossein Karami
# Subject: Learn & Work with MXNet Deep Learning Framework


import mxnet as mx
from mxnet import nd, autograd

# Sample1:
a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
b.asnumpy()
print(b)
print(type(b))
print(type(a))

