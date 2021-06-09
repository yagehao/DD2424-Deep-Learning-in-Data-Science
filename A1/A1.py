# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# author: Yage Hao (yage@kth.se)

import pickle #模块 pickle 实现了对一个 Python 对象结构的二进制序列化和反序列化。
import numpy as np
import matplotlib.pyplot as plt
import unittest
from math import floor, sqrt
from tqdm import tqdm

N = 10000
d = 3072
K = 10
cifar10_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def LoadBatch(filename):
	"""Reads in the data from a CIFAR-10 batch file 
	and returns the image and label data in separate files.

	Args:
		filename (str): filename

	Returns:
		data.T (np.ndarray): image pixel data, size dxn.
					n: number of images, 10000.
					d: dimensionality of each image, 3072 = 32*32*3.
		one_hot_labels.T (np.ndarray): one hot representation of image label, size Kxn.
					K: number of labels, 10.
		labels (np.ndarray): vector of label for each image, each entry is an integer between 0-9, length n.
	"""
	data = np.zeros((N, d))
	labels = np.zeros((N, 1))
	one_hot_labels = np.zeros((N, K))

	dict = unpickle(filename)
	data = dict[bytes("data", 'utf-8')] / 255.0
	labels = np.array(dict[bytes("labels", 'utf-8')])
	one_hot_labels = OneHotEncoding(labels)

	return data.T, one_hot_labels.T, labels

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def OneHotEncoding(labels): 
	"""One-Hot Encoding for categorical variables/nominal"""
	one_hot_labels = np.zeros((N, K))
	for i in range(len(labels)):
		one_hot_labels[i][labels[i]] = 1
	return one_hot_labels


class Classifier():
	def __init__(self, learning_rate, lambda_regularization, n_batch, n_epochs):
		"""Construct a class."""		
		np.random.seed(1)

		self.W = np.zeros((K, d))
		self.b = np.zeros((K, 1))

		self.eta = learning_rate
		self.lambda_reg = lambda_regularization
		self.n_batch = n_batch
		self.n_epochs = n_epochs

		self.initialization()

	def initialization(self):
		mu = 0
		sigma = sqrt(2) / sqrt(d)

		self.W = np.random.normal(mu, sigma, (K, d))
		self.b = np.random.normal(mu, sigma, (K, 1))

	def evaluateClassifier(self, X, W, b):
		"""Compute p=softmax(s).

		Args:
			X (np.ndarray): data matrix, dxn.
			W (np.ndarray): weight matrix, Kxd.
			b (np.ndarray): bias vector, Kx1.
			
		Returns:
			P (np.ndarray): softmax matrix, Kxn,
							each col contains probability of each label 
							for the image in the corresponding col of X.
		"""
		s = np.dot(W, X) + b
		P = self.softmax(s)
		assert(P.shape == (K, X.shape[1]))
		return P

	def softmax(self, x):
		softmax = np.exp(x) / sum(np.exp(x))
		return softmax

	def computeCost(self, X, Y, W, b):
		"""Compute cost function.

		Args:
			X (np.ndarray): data matrix, dxn
			Y (np.ndarray): one hot representation of labels, Kxn
			W (np.ndarray): weight matrix, Kxd.
			b (np.ndarray): bias vector, Kx1.

		Returns:
			final (float): cross-entropy loss
		"""
		regularization = self.lambda_reg * np.sum(np.square(W))
		loss_sum = 0
		for i in range(X.shape[1]):
			x = np.zeros((d, 1))
			y = np.zeros((K, 1))
			x = X[:, [i]]
			y = Y[:, [i]]
			loss_sum += self.cross_entropy(x, y, W=W, b=b)

		loss_sum /= X.shape[1]
		final = loss_sum + regularization
		assert(len(final) == 1)
		return final

	def cross_entropy(self, x, y, W, b):
		l = - np.log(np.dot(y.T, self.evaluateClassifier(x, W=W, b=b)))[0]
		return l

	def ComputeAccuracy(self, X, Y):
		"""Compute the accuracy of the network's predictions.

		Args:
			X (np.ndarray): data matrix, dxn.
			Y (np.ndarray): one-hot labels, Kxn.

		Returns:
			acc (float): accuracy.
		"""
		acc = 0
		for i in range(X.shape[1]):
			P = self.evaluateClassifier(X[:, [i]], self.W, self.b)
			label = np.argmax(P)
			if label == Y[i]:
				acc += 1
		acc /= X.shape[1]
		return acc

	def compute_gradients(self, X, Y, P, W):
		"""Evaluate the gradients of the cost function w.r.t. W and b.

		Args:
			X (np.ndarray): data matrix, dxn.
			Y (np.ndarray): one-hot labels, Kxn.
			W (np.ndarray): weight matrix, Kxd.
			P (np.ndarray): softmax matrix, Kxn,
							each col contains probability of each label 
							for the image in the corresponding col of X.

		Returns:
			the gradient matrix of J w.r.t. W, Kxd.
		"""
		G = -(Y - P.T).T
		return (np.dot(G,X)) / X.shape[0] + 2 * self.lambda_reg * W, np.mean(G, axis=-1, keepdims=True)

	def shuffle(self, a, b):
		"""Shuffle the order of training data before each epoch"""
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def fit(self, X, Y, validationSet=[]):
		"""Model training with mini-batch gradient descent."""
		n = X.shape[1]
		costsTraining = []
		costsValidation = []
		bestW = np.copy(self.W)
		bestb = np.copy(self.b)
		bestVal = self.computeCost(
			validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]
		bestEpoch = 0

		for i in tqdm(range(self.n_epochs)):
			n_batch = floor(n / self.n_batch)

			X, Y = self.shuffle(X.T, Y.T)  
			X = X.T
			Y = Y.T

			for j in range(n_batch):
				j_start = j * self.n_batch
				j_end = (j + 1) * self.n_batch
				if j == n_batch - 1:
					j_end = n

				Xbatch = X[:, j_start:j_end]
				Ybatch = Y[:, j_start:j_end]

				Pbatch = self.evaluateClassifier(Xbatch, self.W, self.b)
				grad_W, grad_b = self.compute_gradients(Xbatch.T, Ybatch.T, Pbatch, self.W)

				self.W -= self.eta * grad_W
				self.b -= self.eta * grad_b

			val = self.computeCost(
				validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]
			print("Validation loss: " + str(val))

			if val < bestVal:
				bestVal = np.copy(val)
				bestW = np.copy(self.W)
				bestb = np.copy(self.b)
				bestEpoch = np.copy(i)
                
			costsTraining.append(self.computeCost(X, Y, self.W, self.b)[0])
			costsValidation.append(val)

		self.W = np.copy(bestW)
		self.b = np.copy(bestb)
		print("Best epoch: " + str(bestEpoch))
		print("Best cost: " + str(self.computeCost(
			validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]))

		plt.plot(costsTraining, label="Training cost")
		plt.plot(costsValidation, label="Validation cost")         
		plt.xlabel('Epoch')
		plt.ylabel('Cost')
		plt.title('Traning &Validation Cost')
		plt.legend(loc='best')
		plt.savefig("training_validation_cost.png")
		plt.show()

		# visualize the weight matrix.
		for i, row in enumerate(self.W):
			img = (row - row.min()) / (row.max() - row.min())
			plt.subplot(2, 5, i + 1)
			img = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
			plt.imshow(img)
			plt.axis('off')
			plt.title(cifar10_labels[i])
		plt.savefig("weights.png")
		plt.show()


def LoadDataset():
	trainSet = {}
	testSet = {}
	validationSet = {}

	for i in [1, 3, 4, 5]:
		t1, t2, t3 = LoadBatch("dataset/data_batch_" + str(i))
		if i == 1:
			trainSet["data"] = t1
			trainSet["one_hot"] = t2
			trainSet["labels"] = t3
		else:
			trainSet["data"] = np.column_stack((trainSet["data"], t1))
			trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], t2))
			trainSet["labels"] = np.append(trainSet["labels"], t3)

	a, b, c = LoadBatch("dataset/data_batch_2")

	#k-fold cross validation
	validationSet["data"], validationSet["one_hot"], validationSet["labels"] = a[:, :1000], b[:, :1000], c[:1000]
	trainSet["data"] = np.column_stack((trainSet["data"], a[:, 1000:]))
	trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], b[:, 1000:]))
	trainSet["labels"] = np.append(trainSet["labels"], c[1000:])
	testSet["data"], testSet["one_hot"], testSet["labels"] = LoadBatch("dataset/test_batch")


	temp = np.copy(trainSet["data"]).reshape((32, 32, 3, 49000), order='F')
	temp = np.flip(temp, 0)
	temp = temp.reshape((3072, 49000), order='F')

	trainSet["data"] = np.column_stack((trainSet["data"], temp))
	trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], trainSet["one_hot"]))
	trainSet["labels"] = np.append(trainSet["labels"], trainSet["labels"])

	mean = np.mean(trainSet["data"], axis=1)
	mean = mean[:, np.newaxis]
	sd = np.std(trainSet["data"])

	trainSet["data"] = (trainSet["data"] - mean)/sd
	validationSet["data"] = (validationSet["data"] - mean)/sd
	testSet["data"] = (testSet["data"] - mean)/sd
	return trainSet, validationSet, testSet


def main():
	print("Loading dataset...")
	trainSet, validationSet, testSet = LoadDataset()
	print("Dataset loaded!")

	lambda_regularization = 1
	n_epochs = 40
	n_batch= 100
	eta = 0.001

	Exercise_1 = Classifier(eta, lambda_regularization, n_batch, n_epochs)
	Exercise_1.fit(trainSet["data"], trainSet["one_hot"], validationSet = validationSet)

	print("lambda=" + str(lambda_regularization) + ",", "n_epochs=" + str(n_epochs) + ",", "n_batch=" + str(n_batch) + ",", "eta=" + str(eta))
	print("Final accuracy:" + str(Exercise_1.ComputeAccuracy(testSet["data"], testSet["labels"])))


if __name__ == "__main__":
	main()