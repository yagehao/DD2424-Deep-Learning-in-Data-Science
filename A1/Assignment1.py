# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# author: Yage Hao (yage@kth.se)

import pickle #模块 pickle 实现了对一个 Python 对象结构的二进制序列化和反序列化。
import numpy as np
import matplotlib.pyplot as plt 

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
	def __init__(self, data, W=None, b=None):
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

	def evaluateClassifier(self, X):
		"""Compute p=softmax(s).

		Args:
			X (np.ndarray): data matrix, dxn.
			
		Returns:
			P (np.ndarray): softmax matrix, Kxn,
							each col contains probability of each label 
							for the image in the corresponding col of X.
		"""
		s = np.dot(self.W, X) + self.b 
		def softmax(s):
			softmax = np.exp(s) / np.sum(np.exp(s))
			return softmax
		P = softmax(s)
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
		l_cross = -np.log(Y*P)
		regul = lam * np.sum(self.W**2)

		J = 1/N * np.sum(l_cross) + regul
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









def main():
	trainX, trainY, trainy = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
	valX, valY, valy = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
	testX, testY, testy = loadBatch("Datasets/cifar-10-batches-py/test_batch")

	trainX = preprocess(trainX)
	valX = preprocess(valX)
	testX = preprocess(testX)

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
	clf = Classifier(data)
	P = clf.evaluateClassifier(trainX[:, :100]) #10x100

if __name__ == "__main__":
	main()