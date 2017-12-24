import random 
import numpy as np
from data_util import load_CIFAR10
import matplotlib.pyplot as plt

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
#store data 
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print("Training data shape: ", X_train.shape)
print("Training labels shape: ", y_train.shape)
print("Test data shape: ", X_test.shape)
print("Test labels shape: ", y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7

for y,cls in enumerate(classes):
	idxs = np.flatnonzero(y_train == y) #y_train == 1 means all data belonging to label 1
#	#select 7 samples from idxs array without replacement i.e no repitition
	idxs = np.random.choice(idxs, samples_per_class, replace=False) 
	for i, idx in enumerate(idxs):
#		#plt_idx is just indexing plot number
		plt_idx = i * num_classes + y + 1
#		#plot a subplot of height 7, width 10, and plt_idx specifies active or not
		plt.subplot(samples_per_class, num_classes, plt_idx)
#		#access X_train array at idx index
		plt.imshow(X_train[idx].astype('uint8'))
		plt.axis('off')
		if i==0:
			plt.title(cls)
plt.show()
"""Reducing the size of dataset to 5000 training samples and 500 test samples"""
num_training = 5000
mask = list(range(5000))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(500))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

from nearest_neighbour import nearest_neighbour

classifier = nearest_neighbour()
classifier.train(X_train, y_train)

dists = classifier.compute_distances_two_loops(X_test)
y_test_pred = classifier.predict_labels(dists, k = 5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print("Got %d / %d correct => accuracy: %f" % (num_correct, num_test, accuracy))

def time_function (f, *args):
	import time
	tic = time.time()
	f(*args)
	toc = time.time()
	return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

dists_one = classifier.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

dists_two = classifier.compute_distances_no_loops(X_test)
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array(np.array_split(X_train, num_folds))
y_train_folds = np.array(np.array_split(y_train, num_folds))

k_to_accuracies = {}
for k in k_choices:
    for n in range(num_folds):
        combinat = [x for x in range(num_folds) if x != n] 
        x_training_dat = np.concatenate(X_train_folds[combinat])
        y_training_dat = np.concatenate(y_train_folds[combinat])
        classifier_k = nearest_neighbour()
        classifier_k.train(x_training_dat, y_training_dat)
        y_cross_validation_pred = classifier_k.predict_labels(X_train_folds[n], k)
        num_correct = np.sum(y_cross_validation_pred == y_train_folds[n])
        accuracy = float(num_correct) / num_test
        k_to_accuracies.setdefault(k, []).append(accuracy)

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print ('k = %d, accuracy = %f' % (k, accuracy))
    print ('mean for k=%d is %f' % (k, np.mean(k_to_accuracies[k])))

for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

best_k = 10

classifier = nearest_neighbour()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)
dists = classifier.compute_distances_no_loops(X_test)
prediction = classifier.predict_labels(dists, k=best_k)
#Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))