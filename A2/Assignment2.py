# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# author: Yage Hao (yage@kth.se)

import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import unittest
import statistics

K = 10
n = 10000
d = 3072
#np.random.seed(100)


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
	#adjust number of hidden layer nodes
	def __init__(self, data, labels, m=50, W1=None, W2=None, b1=None, b2=None):
		"""set up the data structure for the parameters of the network 
		and initialize their values.

		Args:
			data (dict): contains trainX, trainY, trainy, valX, valY, valy, testX, testY, testy
			labels (list): strings of label names
			m (int): number of nodes in hidden layer
			W1 (np.ndarray): weight matrix of layer 1, mxd
			W2 (np.ndarray): weight matrix of layer 2, Kxm
			b1 (np.ndarray): bias vector of layer 1, mx1
			b2 (np.ndarray): bias vector of layer 2, Kx1
		"""
		for key, value in data.items():
			setattr(self, key, value)
		self.labels = labels 
		self.m = m 

		d = self.trainX.shape[0]
		n = self.trainX.shape[1]
		K = len(self.labels)

		if W1 != None:
			self.W1 = W1
		else:
			self.W1 = np.random.normal(0, 1/np.sqrt(d), (m, d))

		if W2 != None:
			self.W2 = W2
		else:
			self.W2 = np.random.normal(0, 1/np.sqrt(m), (K, m))

		if b1 != None:
			self.b1 = b1
		else:
			self.b1 = np.zeros((m, 1))

		if b2 != None:
			self.b2 = b2 
		else:
			self.b2 = np.zeros((K, 1))

	def initialization(self, Wdict, bdict):
		"""A seperate function to initialize the parameters.

		Args:
			Wdict (dict): {Wname: Wshape}.
			bdict (dict): {bname: bshape}.

		Returns:
			new_para (dict): updated W and b.
		"""
		new_para = {}

		for Wname, Wshape in Wdict.items():
			new_para[Wname] = np.random.normal(0, 1/np.sqrt(Wshape[1]), Wshape)

		for bname, bshape in bdict.items():
			new_para[bname] = np.zeros(bshape)

		return new_para

	def relu(self, s1):
		"""ReLU activation, max(0, s1).

		Args:
			s1 (np.ndarray): input

		Returns:
			h (np.ndarray): ReLU result
		"""
		h = s1
		h[h < 0] = 0 #all elements < 0 get are assigned to be 0
		return h

	def softmax(self, s):
		"""Softmax activation.

		Args:
			s (np.ndarray): input

		Returns:
			p (np.ndarray): softmax result
		"""
		#p = np.exp(s) / sum(np.exp(s))
		p = np.exp(s - np.max(s, axis=0)) / \
				np.exp(s - np.max(s, axis=0)).sum(axis=0)
		return p

	def evaluateClassifier(self, X):
		"""Classification function in Figure 1(a).

		Args:
			X (np.ndarray): data matrix, dxn

		Returns:
			h (np.ndarray): ReLU activation results
			p (np.ndarray): Softmax results
		"""
		s1 = self.W1@X + self.b1 
		h = self.relu(s1)
		s = self.W2@h + self.b2 
		p = self.softmax(s)
		return h, p

	def computeCost(self, X, Y, lam):
		"""Cost function in Figure 1(b).

		Args:
			X (np.ndarray): data matrix, dxn
			Y (np.ndarray): one-hot encoding label matrix, Kxn
			lam (float): regularization parameter

		Returns:
			l (float): cross-entropy loss
			J (float): cost
		"""
		n = X.shape[1]
		h, p = self.evaluateClassifier(X)
		l = 1/n * - np.sum(Y*np.log(p))
		J = l + lam * (np.sum(self.W1**2) + np.sum(self.W2**2))
		return l, J

	def computeAccuracy(self, X, y):
		"""Compute accuracy of the classifier.

		Args:
			X (np.ndarray): data matrix, dxn.
			y (np.ndarray): label vector, nx1.

		Returns:
			acc (float): model accuracy.
		"""
		pred = np.argmax(self.evaluateClassifier(X)[1], axis=0) #max p
		acc = pred.T[pred == np.asarray(y)].shape[0] / X.shape[1]
		return acc 

	def computeGradient(self, Xbatch, Ybatch, lam):
		"""Compute gradient of J w.r.t. W and b analytically.
		Ref: Lecture 4 slides.

		Args:
			Xbatch (np.ndarray): data (batch) matrix, dxn
			Ybatch (np.ndarray): one-hot encoding label (batch) matrix, Kxn
			lam (float): regularization parameter

		Returns:
			W1grad (np.ndarray): gradient of J w.r.t. W1
			W2grad (np.ndarray): gradient of J w.r.t. W2
			b1grad (np.ndarray): gradient of J w.r.t. b1
			b2grad (np.ndarray): gradient of J w.r.t. b2
		"""
		nb = Xbatch.shape[1]

		#forward pass
		Hbatch, Pbatch = self.evaluateClassifier(Xbatch)

		#backward pass
		Gbatch = -(Ybatch - Pbatch)

		W2grad = 1/nb * Gbatch@Hbatch.T + 2*lam*self.W2 
		b2grad = 1/nb * Gbatch@np.ones(nb)
		b2grad = np.reshape(b2grad, (Ybatch.shape[0], 1))

		Gbatch = self.W2.T@Gbatch
		Hbatch[Hbatch <= 0] = 0
		Gbatch = np.multiply(Gbatch, Hbatch>0)

		#W1grad = 1/nb * Gbatch@Xbatch.T + lam*self.W1
		W1grad = 1/nb * Gbatch@Xbatch.T + 2*lam*self.W1
		b1grad = 1/nb * Gbatch@np.ones(nb)
		b1grad = np.reshape(b1grad, (self.m, 1))

		return W1grad, W2grad, b1grad, b2grad

	def computeGradientNum(self, Xbatch, Ybatch, lam=0, margin=1e-5):
		"""Compute gradient numerically to check correctness.
		
		Args:
			Xbatch (np.ndarray): data batch matrix.
			Ybatch (np.ndarray): one-hot label batch matrix.
			lam (float): regularizaiton parameter, set to 0 for check.
			margin (float): margin to assert equality.

		Results:
			grads (dict): include numerical results of W1grad, W2grad, b1grad, b2grad
		"""
		grads = {}
		for j in range(1, 3):
			selfW = getattr(self, 'W' + str(j))
			selfB = getattr(self, 'b' + str(j))
			grads['W' + str(j)] = np.zeros(selfW.shape)
			grads['b' + str(j)] = np.zeros(selfB.shape)

			b_try = np.copy(selfB)
			for i in range(selfB.shape[0]):
				selfB = b_try[:]
				selfB[j] = selfB[j] + margin
				_, c2 = self.computeCost(Xbatch, Ybatch, lam)
				getattr(self, 'b' + str(j))[i] = getattr(self, 'b' + str(j))[i] - 2*margin
				_, c3 = self.computeCost(Xbatch, Ybatch, lam)
				grads['b' + str(j)][i] = (c2-c3) / (2*margin)

			W_try = np.copy(selfW)
			for i in np.ndindex(selfW.shape):
				selfW = W_try[:,:]
				selfW[i] = selfW[i] + margin
				_, c2 = self.computeCost(Xbatch, Ybatch, lam)
				getattr(self, 'W' + str(j))[i] = getattr(self, 'W' + str(j))[i] - 2*margin
				_, c3 = self.computeCost(Xbatch, Ybatch, lam)
				grads['W' + str(j)][i] = (c2-c3) / (2*margin)

		return grads['W1'], grads['W2'], grads['b1'], grads['b2']

	def minibatchGD(self, X, Y, lam=0.01, batchsize=100, 
		eta_min=1e-5, eta_max=1e-1, n_s=500, epochs=10,
		plot=True, text=True):
		"""train the network using mini-batch gradient descent 
		with cyclical learning rates.

		Args:
			X (np.ndarray): data matrix, dxn
			Y (np.ndarray): one-hot encoding label matrix, Kxn
			lam (float): regularization parameter
			batchsize (int): batch size
			eta_min (float): lower bound of learning rate eta
			eta_max (float): upper bound of learning rate eta
			n_s (int): stepsize, one complete cycle will take 2n_s update steps
			epochs (int): number of training epochs
			plot (bool): whether to plot costs
			text (bool): whether to output text
		Returns:
			trainAcc (float): training accuracy
			valAcc (float): validation accuracy
			testAcc (float): testing accuracy
		"""
		num_batch = int(np.floor(X.shape[1]/batchsize)) #向下取整
		eta = eta_min #initial eta
		t = 0

		cost_train_ls = np.zeros(epochs)
		cost_val_ls = np.zeros(epochs)
		loss_train_ls = np.zeros(epochs)
		loss_val_ls = np.zeros(epochs)
		acc_train_ls = np.zeros(epochs)
		acc_val_ls = np.zeros(epochs)


		for i in range(epochs):
			for j in range(num_batch):
				#divide whole dataset into data batches
				N = int(X.shape[1]/num_batch)
				j1 = j * N 
				j2 = (j+1) * N
				Xbatch = X[:, j1:j2]
				Ybatch = Y[:, j1:j2]

				W1grad, W2grad, b1grad, b2grad = self.computeGradient(Xbatch, Ybatch, lam)

				self.W1 -= eta * W1grad
				self.W2 -= eta * W2grad
				self.b1 -= eta * b1grad
				self.b2 -= eta * b2grad

				#cyclic learning rate
				if t <= n_s:
					eta = eta_min + t * (eta_max - eta_min) / n_s
				elif t <= 2*n_s:
					eta = eta_max - (t - n_s) * (eta_max - eta_min) / n_s 
				t = (t+1) % (2*n_s)

			if plot==True:
				loss_train_ls[i], cost_train_ls[i] = self.computeCost(X, Y, lam)
				loss_val_ls[i], cost_val_ls[i] = self.computeCost(self.valX, self.valY, lam)
				acc_train_ls[i] = self.computeAccuracy(self.trainX, self.trainy)
				print(acc_train_ls[i])
				acc_val_ls[i] = self.computeAccuracy(self.valX, self.valy)

		trainAcc = self.computeAccuracy(self.trainX, self.trainy)
		valAcc = self.computeAccuracy(self.valX, self.valy)
		testAcc = self.computeAccuracy(self.testX, self.testy)

		if text==True:
			print("training accuracy:", str(trainAcc))
			print("validation accuracy:", str(valAcc))
			print("testing accuracy:", str(testAcc))

		if plot==True:
			def fplot(xvalue, yvalue1, yvalue2, title, y_label, y_min, y_max):
				fig, ax = plt.subplots(figsize = (10, 8))
				ax.plot(xvalue, yvalue1, label='training')
				ax.plot(xvalue, yvalue2, label='validation')
				ax.legend()
				ax.set(xlabel='Number of epochs', ylabel=y_label)
				ax.set_ylim([y_min, y_max])
				ax.grid()
				plt.show()

			fplot(np.arange(epochs), cost_train_ls, cost_val_ls, 'Cost Plot', 'Cost', 0, 4)
			fplot(np.arange(epochs), loss_train_ls, loss_val_ls, 'Loss Plot', 'Loss', 0, 3)
			fplot(np.arange(epochs), acc_train_ls, acc_val_ls, 'Accuracy Plot', 'Accuracy', 0, 1)

		return trainAcc, valAcc, testAcc


class TestEqualityMethods(unittest.TestCase):
	def assertEqual(self, array1, array2, dec=4):
		"""Assert two arrays are equal in 4 decimal places."""
		np.testing.assert_almost_equal(array1, array2, decimal=dec)

	def testEquality(self):
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

		clf = Classifier(data, labels)

		Wdict = {'W1': (50, 20)}
		bdict = {}
		new_para = clf.initialization(Wdict, bdict)

		clf.W1 = new_para['W1']
		anaW1grad, anaW2grad, anab1grad, anab2grad = \
			clf.computeGradient(clf.trainX[:20, :2], clf.trainY[:20, :2], lam=0)
		numW1grad, numW2grad, numb1grad, numb2grad = \
			clf.computeGradientNum(clf.trainX[:20, :2], clf.trainY[:20, :2], lam=0)

		self.assertEqual(anaW1grad, numW1grad)
		self.assertEqual(anaW2grad, numW2grad)
		self.assertEqual(anab1grad, numb1grad)
		self.assertEqual(anab2grad, numb2grad)


def replicate():
	#read in data
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

	clf = Classifier(data, labels)
	clf.minibatchGD(data['trainX'], data['trainY'], lam=0.01,
			batchsize=100, eta_min=1e-5, eta_max=1e-1, n_s=500, epochs=10,
			plot=True, text=True) #replicate figure 3
	clf.minibatchGD(data['trainX'], data['trainY'], lam=0.01,
			batchsize=100, eta_min=1e-5, eta_max=1e-1, n_s=800, epochs=48,
			plot=True, text=True) #replicate figure4

def fit(valsize=5000):
	"""fit the best classifier.

	Args:
		valsize (int): size of the validation set.
	"""
	#read in all 5 batches of data
	trainX1, trainY1, trainy1 = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
	trainX2, trainY2, trainy2 = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
	trainX3, trainY3, trainy3 = loadBatch("Datasets/cifar-10-batches-py/data_batch_3")
	trainX4, trainY4, trainy4 = loadBatch("Datasets/cifar-10-batches-py/data_batch_4")
	trainX5, trainY5, trainy5 = loadBatch("Datasets/cifar-10-batches-py/data_batch_5")
	testX, testY, testy = loadBatch("Datasets/cifar-10-batches-py/test_batch")

	trainX = np.concatenate((trainX1, trainX2, trainX3, trainX4, trainX5), axis=1)
	trainY = np.concatenate((trainY1, trainY2, trainY3, trainY4, trainY5), axis=1)
	trainy = np.concatenate((trainy1, trainy2, trainy3, trainy4, trainy5))

	valX = trainX[:, -valsize:]
	valY = trainY[:, -valsize:]
	valy = trainy[-valsize:]

	trainX = trainX[:, :-valsize]
	trainY = trainY[:, :-valsize]
	trainy = trainy[:-valsize]

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

	acc_train_set = []
	acc_val_set = []
	acc_test_set = []
	for i in range(1):
		print(i)
		clf = Classifier(data, labels)
		trainAcc, valAcc, testAcc = clf.minibatchGD( #adjust parameters
			data['trainX'], data['trainY'], lam=0.002,
			batchsize=100, eta_min=1e-5, eta_max=1e-1, n_s=500,
			epochs=20, text=False, plot=True) #two cycle
		acc_train_set.append(trainAcc)
		acc_val_set.append(valAcc)
		acc_test_set.append(testAcc)

	print('training accuracy:', statistics.mean(acc_train_set))
	print('validation accuracy:', statistics.mean(acc_val_set))
	print('testing accuracy:', statistics.mean(acc_test_set))
	#print('standard deviation of training accuracy:', statistics.stdev(acc_train_set))
	#print('standard deviation of validation accuracy:', statistics.stdev(acc_val_set))
	#print('standard deviation of testing accuracy:', statistics.stdev(acc_test_set))

if __name__ == "__main__":

	#replicate()

	#fit(valsize=1000) #adjust size of validation set

	#test numerically and analytically computed gradients
	unittest.main()