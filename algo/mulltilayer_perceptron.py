from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math


EPOCHS = 500
LR = 2

activate = np.vectorize(lambda x: 1/( 1+math.exp(-x) ) )
#activate = np.vectorize( lambda x : 1 if x>0.5 else 0 )


def predict( I, H ):

	A = []
	input = I
	for h in H:
		A.append( activate( np.dot(input, h) ) )
		input = A[-1]

	return A


if __name__ == "__main__":

	I = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	T = np.array([[0], [0], [0], [1]])

	H = []
	H.append( np.array( [[(random()-0.5)*0.2 for x in range(1)]]*I.shape[1] ) )
	#print H

	for _ in range(EPOCHS):

		A = predict(I, H)

		e = T-A[-1]
		print "\nerror : ", e
		for i in range(len(H)):
			input = A[-(i+2)] if i+1<len(H) else I
			a = A[-(i+1)]
			h = H[-(i+1)]
			e = e

			print "\n\n", "e : ", e, "\n a : ", a, "\n h : ", h, "\ninput :", input
			print "\nlr : ", LR*np.dot( np.transpose(input), e )

			H[-(i+1)] = h+LR*np.dot( np.transpose(input), e )
			
			print "\n h after : ", H[-(i+1)]
			e = np.dot( e, np.transpose(h) )
		
		
		print "\n\n"
	
	print predict( I, H )[-1]
		