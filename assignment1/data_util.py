from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def load_pickle(f):
	version = platform.python_version_tuple()
	if version[0] == '2':
		return pickle.load(f)
	elif version[0] == '3':
		return pickle.load(f, encoding='latin1') #deserializing of data i.e conversion of bytes to obj
	raise ValueError('Invalid python version: {}'.format(version))

def load_CIFAR_batch(filename):
	"""load single batch of cifar"""
	#load file in read binary mode
	with open(filename, 'rb') as f:
		#save file in dictionary
		datadict = load_pickle(f)
		X = datadict['data'] #save data in X label
		Y = datadict['labels'] #save data in Y label
		#reshape and transpose 1D matrix to 4D matrix of size 10000 values (images) of 32 tuples (height) of 32 rows (width) & 3 columns (rgb values) each
		X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
		Y = np.array(Y)
		return X,Y

def load_CIFAR10(ROOT):
	"""load cifar 10 data as obj from binary file stream"""
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, 'data_batch_%d' % (b, )) #joins path
		X,Y = load_CIFAR_batch(f)
		xs.append(X)
		ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X,Y
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
	return Xtr,Ytr,Xte,Yte