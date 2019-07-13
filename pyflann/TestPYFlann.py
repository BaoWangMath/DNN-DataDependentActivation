#!/usr/bin/python
"""
This code is used to test the python flann for nearest neighbor searching.
"""
from __future__ import absolute_import
import numpy as np
import keras
from keras.datasets import mnist
from scipy import spatial
import time
from pyflann import *

# Load MNIST data
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('Number of training examples: ', x_train.shape[0])
print('Number of testing examples: ', x_test.shape[0])
print('The shape of the training labels: ', y_train.shape)
print('The shape of the testing labels: ', y_test.shape)

fea = np.zeros((len(x_train)+len(x_test), 784))
gnd = np.zeros((len(x_train)+len(x_test),))
fea[:60000, :] = x_train; fea[60000:, :] = x_test
gnd[:60000] = y_train; gnd[60000:] = y_test
gnd[gnd == 0] = 10

# Select a subset of the data to save time
n = 1000
fea = fea[:n, :]
gnd = gnd[:n]

flann = FLANN()

start = time.time()
result, dists = flann.nn(fea, fea, 10, algorithm="kdtree", branching=32, iterations=7, checks=16)
end = time.time()
print('Results: ', result)
print('Dist: ', dists)
print('Time: ', end-start)
"""
dataset = np.array(
    [[1., 1, 1, 2, 3],
     [10, 10, 10, 3, 2],
     [100, 100, 2, 30, 1]
     ])
testset = np.array(
    [[1., 1, 1, 1, 1],
     [90, 90, 10, 10, 1]
     ])
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
print result
print dists

dataset = np.random.rand(10000, 128)
testset = np.random.rand(1000, 128)
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
print result
print dists
"""
