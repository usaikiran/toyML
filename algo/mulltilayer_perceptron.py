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

		A = predict(I, W)
		e = T-A[-1]

		verror = round(np.sum( square(e) ), 3)
		print "epoch", epoch, " validation error :", verror
		if verror<0.001:
			print "early stopping"
			break

		for x in range(1, len(W)+1):

			i = A[-(x+1)] if x<len(W) else I
			a = A[-x]
			w = W[-x]
			e = e*(1-a)*a

			W[-x] += LR*np.dot( np.transpose(i), e )
			e = np.dot( e, np.transpose(w) )
	
	return W


if __name__ == "__main__":

	I = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
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
