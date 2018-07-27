# Amir Hossein Karami
# Subject: Learn & Work with MXNet Deep Learning Framework
# Main Link: http://mxnet.incubator.apache.org/tutorials/python/predict_image.html
# Subject: recognize objects in an image with a pre-trained model & feature extraction


import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import namedtuple
# from mxnet import nd, autograd


# Load & Download the ResNet152 pre-trained Model:

# path='http://data.mxnet.io/models/imagenet-11k/'
# [mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json'),
#  mx.test_utils.download(path+'resnet-152/resnet-152-0000.params'),
#  mx.test_utils.download(path+'synset.txt')]


# Load the ResNet152 trained of the Full ImageNet:

sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


