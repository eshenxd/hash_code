#!/usr/bin/python

import numpy as np
from scipy.io import loadmat, savemat
import sys

args = sys.argv
if len(args) < 2:
  num = 500
else:
  num = int(args[1])

loopbits = [8, 16, 24, 32, 48, 64]

trainset = loadmat('../cifar10_train.mat')
testset = loadmat('../cifar10_test.mat')
trainY = trainset['labels']
testY = testset['labels']

for r in loopbits:
  dist_fn = "ksh_dist_%dbit_50K.mat" % r
  dataset = loadmat(dist_fn)
  data = dataset['dist']
  del dataset
  ntest = data.shape[0]
  sindex = data.argsort(axis=1)
  pindex = np.array(xrange(ntest))
  pindex.shape = ntest, 1

  sorted_data = data[pindex, sindex]

  precision = []
  for i in xrange(ntest):
    pos = num
    if pos != 0:
      candindex = sindex[i][:pos]
      y = testY[i]
      retY = trainY[candindex]
      ncorrect = (retY == y).sum()
      # precision.append(float(ncorrect)/float(pos))
      precision.append(float(ncorrect)/5000.0)
    else:
      precision.append(0)
  print "%d bit, precision: %f" % (r, np.mean(precision))
  
