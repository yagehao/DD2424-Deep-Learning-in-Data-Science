# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# author: Yage Hao (yage@kth.se)

import pickle #模块 pickle 实现了对一个 Python 对象结构的二进制序列化和反序列化。
import numpy as np
import matplotlib.pyplot as plt
import unittest
import statistics
import re 

K = 10
n = 10000
d = 3072


def loadBatch(filename):
	"""Reads in the data from a CIFAR-10 batch file 
	and returns the image and label data in separate files.

	Args:
		filename (str): filename

	Returns:
		X (np.ndarray): image pixel data, size dxn.
					n: number of images, 10000.
					d: dimensionality of each image, 3072 = 32*32*3.
		Y (np.ndarray): one hot representation of image label, size Kxn.
					K: number of labels, 10.
		y (np.ndarray): vector of label for each image, each entry is an integer between 0-9, length n.
	"""
	with open(filename, 'rb') as f:
		dataDict = pickle.load(f, encoding='bytes') #从已打开的 file object 文件中读取封存后的对象，
													#重建其中特定对象的层次结构并返回。
		X = (dataDict[bytes('data', 'utf-8')] / 255.0).T #convert entries to values between 0 and 1
		y = np.array(dataDict[bytes('labels', 'utf-8')])
		Y = (np.eye(10)[y]).T #one-hot vector conversion
							  #https://stackoverflow.com/questions/45068853/how-does-this-one-hot-vector-conversion-work
	return X, Y, y

def unpickle(filename):
	with open(filename, 'rb') as f:
		file_dict = pickle.load(f, encoding='bytes')
	return file_dict

def preprocess(X):
	"""Standarization.

	Args:
		X (np.ndarray): image pixel data, size dxn.

	Returns:
		X (np.ndarray): standarized X.
	"""
	mean_X = np.mean(X, axis=1) #1xd
	mean_X = mean_X[:, np.newaxis] #dx1

	std_X = np.std(X, axis=1) #1xd
	std_X = std_X[:, np.newaxis] #dx1

	X = (X - mean_X)/std_X #dxn: 3072x10000
	return X


class Classifier():
	def __init__(self, data, labels, W=None, b=None):
		"""Construct a class.

		Args:
			data (dict):
                - training, validation and testing:
                    - examples matrix
                    - one-hot-encoded labels matrix
                    - labels vector
			W (np.ndarray): model weight parameters, Kxd.
			b (np.ndarray): model bias parameters, Kx1.
		"""		
		np.random.seed(100)
		self.W = np.random.normal(0, 0.01, (K, d)) #initialize model parameters
		self.b = np.random.normal(0, 0.01, (K, 1))

		for k, v in data.items():
			setattr(self, k, v)

		self.labels = labels


	def evaluateClassifier(self, X):
		"""Compute p=softmax(s).

		Args:
			X (np.ndarray): data matrix, dxn.
			
		Returns:
			P (np.ndarray): softmax matrix, Kxn,
							each col contains probability of each label 
							for the image in the corresponding col of X.
		"""
		s = self.W@X + self.b 
		P = np.exp(s - np.max(s, axis=0)) / np.exp(s - np.max(s, axis=0)).sum(axis=0)
		return P


	def computeCost(self, X, Y, lam):
		"""Compute cost function.

		Args:
			X (np.ndarray): data matrix, dxn
			Y (np.ndarray): one hot representation of labels, Kxn
			lam (float): regularization term

		Returns:
			J (float): cross-entropy loss
		"""
		N = X.shape[1]

		P = self.evaluateClassifier(X)
		J = 1/N * - np.sum(Y*np.log(P)) + lam * np.sum(self.W**2)
		return J


	def computeAccuracy(self, X, y):
		"""Compute the accuracy of the network's predictions.

		Args:
			X (np.ndarray): data matrix, dxn.
			y (np.ndarray): vector of labels, n.

		Returns:
			acc (float): accuracy.
		"""
		pred = np.argmax(self.evaluateClassifier(X), axis=0)

		pred_true = pred.T[pred == np.asarray(y)].shape[0] #https://stackoverflow.com/questions/48134598/x-shape0-vs-x0-shape-in-numpy

		acc = pred_true/X.shape[1]
		return acc


	def computeGradients(self, Xbatch, Ybatch, lam):
		"""Evaluate the gradients of the cost function w.r.t. W and b.

		Args:
			Xbatch (np.ndarray): data matrix, dxn.
			Ybatch (np.ndarray): one-hot labels, Kxn.
			lam (float): regularization term.

		Returns:
			grad_W (np.ndarray): the gradient matrix of J w.r.t. W, Kxd.
			grad_b (np.ndarray): the gradient vector of J w.r.t. b, Kx1.
		"""
		N = Xbatch.shape[1]
		Pbatch = self.evaluateClassifier(Xbatch)
		Gbatch = -(Ybatch - Pbatch)

		grad_W = 1/N * Gbatch@Xbatch.T + 2 * lam * self.W 
		grad_b = np.reshape(1/N * Gbatch@np.ones(N), (Ybatch.shape[0], 1))
		return grad_W, grad_b


	def computeGradientsNum(self, Xbatch, Ybatch, lam=0, h=1e-6):
		"""Numerically evaluate gradients.

		Args:
			Xbatch (np.ndarray): data matrix, dxn.
			Ybatch (np.ndarray): one hot labels, Kxn.
			lam (float): regularization term.
			h (float): marginal offset.

		Returns:
			grad_W (np.ndarray): the gradient matrix of J w.r.t. W, Kxd.
			grad_b (np.ndarray): the gradient vector of J w.r.t. b, Kx1.
		"""
		grad_W = np.zeros(self.W.shape)
		grad_b = np.zeros(self.b.shape)

		b_dummy = np.copy(self.b)
		for i in range(len(self.b)):
			self.b = b_dummy
			self.b[i] = self.b[i] + h
			c2 = self.computeCost(Xbatch, Ybatch, lam)
			self.b[i] = self.b[i] - 2*h
			c3 = self.computeCost(Xbatch, Ybatch, lam)
			grad_b[i] = (c2-c3) / (2*h)

		W_dummy = np.copy(self.W)
		for i in np.ndindex(self.W.shape):
			self.W = W_dummy
			self.W[i] = self.W[i] + h
			c2 = self.computeCost(Xbatch, Ybatch, lam)
			self.W[i] = self.W[i] - 2*h
			c3 = self.computeCost(Xbatch, Ybatch, lam)
			grad_W[i] = (c2-c3) / (2*h)

		return grad_W, grad_b


	def performPlot(self, n_epochs, trainCost, valCost):
			"""Plot performance curve of training set and validation set."""
			epochs = np.arange(n_epochs)
			fig, ax = plt.subplots(figsize = (8,8))
			ax.plot(epochs, trainCost, label="training")
			ax.plot(epochs, valCost, label="validation")
			ax.legend()
			ax.set(xlabel='Num of epochs', ylabel='costs')
			ax.grid()
			plt.savefig("Result_Pics/performance.png", bbox_inches='tight')


	def minibatchGD(self, X, Y, lam=0, n_batch=100, eta=0.1, n_epochs=20, performPlot=False, text=False):
		"""Model training with mini-batch gradient descent.

		Args:
			X (np.ndarray): data matrix, DxN
			Y (np.ndarray): one hot representing label matrix, KxN
			lam (float): regularization term.
			n_batch (int): size of mini batches.
			eta (float): learning rate.
			n_epochs (int): number of runs through the whole training set.
			performPlot (bool): decide whether to plot costs.
			text (bool): decide whether to output

		Returns:
			trainAccu (float): accuracy of the training set.
			valAccu (float): accuracy of the validation set.
			testAccu (float): accuracy of the test set.
		"""
		if performPlot==True:
			trainCost = np.zeros(n_epochs)
			valCost = np.zeros(n_epochs)

		for epo in range(n_epochs):
			for j in range(n_batch):
				N = int(X.shape[1] / n_batch)

				j_start = j * N
				j_end = (j+1) * N

				Xbatch = X[:, j_start:j_end]
				Ybatch = Y[:, j_start:j_end]

				grad_W, grad_b = self.computeGradients(Xbatch, Ybatch, lam)
				self.W -= eta * grad_W
				self.b -= eta * grad_b

			if performPlot == True:
				trainCost[epo] = self.computeCost(X, Y, lam)
				valCost[epo] = self.computeCost(self.valX, self.valY, lam)

		if performPlot == True:
			self.performPlot(n_epochs, trainCost, valCost)

		trainAccu = self.computeAccuracy(self.trainX, self.trainy)
		valAccu = self.computeAccuracy(self.valX, self.valy)
		testAccu = self.computeAccuracy(self.testX, self.testy)

		if text == True:
			print("training accuracy = " + str(trainAccu))
			print("validation accuracy = " + str(valAccu))
			print("testing accuracy = " + str(testAccu))

		return trainAccu, valAccu, testAccu


	def visualization(self, plotNum, save=False):
		"""visualize the weight matrix.

		Args:
			plotNum (int): number of plots.
			save (bool): whether save the images.
		"""
		for i,j in enumerate(self.W):
			j = (j - np.min(j)) / (np.max(j) - np.min(j))

			img = np.dstack((
				j[0:1024].reshape(32, 32),
				j[1024:2048].reshape(32, 32),
				j[2048:].reshape(32, 32)
				))

			title = re.sub('b\'', '', str(self.labels[i]))
			title = re.sub('\'', '', title)
			fig = plt.figure(figsize=(3, 3))
			ax = fig.add_subplot(111)
			ax.imshow(img, interpolation='bicubic')
			ax.set_title('Category = ' + title, fontsize=15)

			if save==True:
				plt.savefig("Result_Pics/" + str(num) + "_" + title + ".png", bbox_inches="tight")

			plt.show()


def main():
	trainX, trainY, trainy = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
	valX, valY, valy = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
	testX, testY, testy = loadBatch("Datasets/cifar-10-batches-py/test_batch")

	trainX = preprocess(trainX)
	valX = preprocess(valX)
	testX = preprocess(testX)

	labels = unpickle('Datasets/cifar-10-batches-py/batches.meta')[ b'label_names']

	data = {
		'trainX': trainX,
		'trainY': trainY, 
		'trainy': trainy,
		'valX': valX,
		'valY': valY,
		'valy': valy,
		'testX': testX,
		'testY': testY,
		'testy': testy
	}

	#q4: check function run
	clf = Classifier(data, labels)
	P = clf.evaluateClassifier(trainX[:, :100]) #10x100
	print("P computing completed!")

	lams = [0, 0, 0.1]
	etas = [0.1, 0.01, 0.01] 

	for i in range(3):
		print("i =", i)

		trainAccuSet = []
		valAccuSet = []
		testAccuSet = []

		for j in range(10):
			print("j = ", j)

			trainAccu, valAccu, testAccu = clf.minibatchGD(
				trainX, trainY, lam=lams[i], eta=etas[i])

			trainAccuSet.append(trainAccu)
			valAccuSet.append(valAccu)
			testAccuSet.append(testAccu)

		print("lam = ", lams[i], "eta = ", etas[i], ':\n')
		print("training accuracy:", trainAccuSet, '\n')
		print("validation accuracy:", valAccuSet, '\n')
		print("testing accuracy:", testAccuSet, '\n')

		clf.visualization(plotNum=i)

if __name__ == "__main__":
	main()