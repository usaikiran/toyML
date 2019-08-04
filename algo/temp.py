'''
- A simple multilayer perceptron with fixed no of hidden layers
'''

from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math

 	
activate = np.vectorize(lambda x: 1/( 1+math.exp(-x) ) )
derivative = lambda x : x*( 1-x )

def predict(  ):
	pass


input_vector = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
target_vector = np.array( [ [0], [0], [0], [1] ] )

hidden_weights = [ [ ( (random()-0.5)*2 )*( 1/math.sqrt(input_vector.shape[1]) ) ]*2 ]*input_vector.shape[1]
output_weights = [ [ ( (random()-0.5)*2 )*( 1/math.sqrt(2) ) ]*1 ]*2

layer_1_output = np.dot( input_vector, hidden_weights )
layer_2_input = activate( layer_1_output )

#print layer_2_input
layer_2_output = activate( np.dot( layer_2_input, output_weights ) )

#print layer_2_output
layer_2_error = target_vector-layer_2_output
print target_vector, layer_2_output, layer_2_error