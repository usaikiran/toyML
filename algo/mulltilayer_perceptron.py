
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

round = lambda x,n : int(x*(10**n)) / (10**n)

def predict( I, H ):

	A = []
	input = I
	for h in H:
		A.append( activate( np.dot(input, h)+BIAS ) )
		input = A[-1]

	return A


def train( I, T, W ):

	for epoch in range(EPOCHS):

		# activated outputs of all layers
		A = predict(I, W)

		# error at final layer
		e = T-A[-1]

		# validation error
		verror = round(np.sum( square(e) ), 3)
		print "epoch", epoch, " validation error :", verror
		if verror<0.001:
			print "early stopping"
			break

		for x in range(1, len(W)+1):

			# i - input to the layer
			i = A[-(x+1)] if x<len(W) else I
			# o - output from the layer
			o = A[-x]
			# w - weights of the layer
			w = W[-x]
			# e - error of the layer
			e = e*(1-o)*o

			# updating weights of the layer
			W[-x] += LR*np.dot( np.transpose(i), e )
			# error for next layer
			e = np.dot( e, np.transpose(w) )
	
	return W


if __name__ == "__main__":

	# input to mlp
	I = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	# expected output
	T = np.array([[1], [0], [0], [1]])

	layers = [3, 3]
	layers.append(T.shape[1])
	W = []

	inputNodes = I.shape[1]
	for w in layers:
		W.append( np.array( [ [ ( random()-0.5)*0.2 for x in range(w) ] for x in range(inputNodes) ] ) )
		inputNodes = w

	W = train( I, T, W )
		
	print predict( I, W )[-1]
