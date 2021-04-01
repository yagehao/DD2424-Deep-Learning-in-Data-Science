# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# author: Yage Hao (yage@kth.se)

import pickle #模块 pickle 实现了对一个 Python 对象结构的二进制序列化和反序列化。
import numpy as np
import matplotlib.pyplot as plt 


def LoadBatch(filename):
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

def Preprocess(X):
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




def main():
	Xtrain, Ytrain, ytrain = LoadBatch("Datasets/cifar-10-batches-py/data_batch_1")
	Xval, Yval, yval = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
	Xtest, Ytest, ytest = LoadBatch("Datasets/cifar-10-batches-py/test_batch")

	Xtrain = Preprocess(Xtrain)
	Xval = Preprocess(Xval)
	Xtest = Preprocess(Xtest)

if __name__ == "__main__":
	main()