
from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math


EPOCHS = 20000
LR = 0.9
BIAS = -1

activate = np.vectorize(lambda x: 1/( 1+math.exp(-x) ) )
square = np.vectorize( lambda x: x*x )


class MultilayerPerceptron:

	def __init__( self, inpDim, targetVector ):

		self.inpDim = inpDim
		self.T = targetVector

		self.__generateWeights__()

	
	def __generateWeights__(self):

		layers = [3, 3]
		layers.append(self.T.shape[1])
		self.W = []

		inputNodes = self.inpDim
		for w in layers:
			self.W.append( np.array( [ [ ( random()-0.5)*0.2 for x in range(w) ] for x in range(inputNodes) ] ) )
			inputNodes = w


	def predict( self, I ):

		A = []
		input = I
		for h in self.W:
			A.append( activate( np.dot(input, h)+BIAS ) )
			input = A[-1]

		return A


	def train( self, I ):

		self.__generateWeights__()

		for epoch in range(EPOCHS):

			# activated outputs of all layers
			A = self.predict(I)

			# error at final layer
			e = self.T-A[-1]

			# validation error
			verror = round(np.sum( square(e) ), 3)

			if __name__ == "__main__":
				print "epoch", epoch, " validation error :", verror

			if verror<0.001:
				if __name__ == "__main__":
					print "early stopping"
				break

			for x in range(1, len(self.W)+1):

				# i - input to the layer
				i = A[-(x+1)] if x<len(self.W) else I
				# o - output from the layer
				o = A[-x]
				# w - weights of the layer
				w = self.W[-x]
				# e - error of the layer
				e = e*(1-o)*o

				# updating weights of the layer
				self.W[-x] += LR*np.dot( np.transpose(i), e )
				# error for next layer
				e = np.dot( e, np.transpose(w) )
		


if __name__ == "__main__":

	# input to mlp
	I = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	# expected output
	T = np.array([[1], [0], [0], [0]])

	perceptron = MultilayerPerceptron( 2, T )
	perceptron.train( I )
	print perceptron.predict( I )[-1]