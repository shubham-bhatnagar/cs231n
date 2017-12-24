import numpy as np
import matplotlib.pyplot as plt

class nearest_neighbour(object):
    
	def __init__(self):
		pass
	
	def train(self, X, y):
		#store training data
		self.X_train = X
		self.y_train = y
	
	def predict(self, X, k=1, num_loops=0):
		if num_loops == 0:
			dists = self.compute_distances_no_loops(X)
		elif num_loops == 1:
			dists = self.compute_distances_one_loop(X)
		elif num_loops == 2:
			dists = self.compute_distances_two_loops(X)
		else:
			raise ValueError('Invalid value %d for num_loops' % num_loops)
		return self.predict_labels(dists, k=k)

	def compute_distances_two_loops(self, X):
		num_train = self.X_train.shape[0]
		num_test = X.shape[0]
		dists = np.zeros((num_test, num_train))
		for i in range(num_test):
			for j in range(num_train):
				dists [i,j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
		return dists
	
	def compute_distances_one_loop (self, X):
		num_train = self.X_train.shape[0]
		num_test = X.shape[0]
		dists = np.zeros((num_test, num_train))
		for i in range(num_test):
			#store the dist of ith test data point with every other point in training data
			dists [i] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis = 1)) #axis = 1 means horizontal addition
		return dists

	def compute_distances_no_loops (self, X):
		num_train = self.X_train.shape[0]
		num_test = X.shape[0]
		dists = np.zeros((num_test, num_train))
		dists = np.sqrt((X ** 2).sum(axis = 1, keepdims = True) + (self.X_train ** 2).sum(axis = 1) - 2 * X.dot(self.X_train.T))
		return dists

	def predict_labels (self, dists, k = 1):
		classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		num_test = dists.shape[0]
		y_pred = np.zeros((num_test))
		for i in range(num_test):
			closest_y = []
			closest_y = self.y_train[np.argsort(dists[i])][:k]
			y_pred[i] = np.argmax(np.bincount(closest_y))
		return y_pred