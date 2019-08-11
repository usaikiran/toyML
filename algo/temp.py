'''
- A simple multilayer perceptron with fixed no of hidden layers
'''

from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math

 	
sigmoid = np.vectorize( lambda x: 1/( 1+math.exp(-x) ) )
square = np.vectorize( lambda x: x*x )

derivative = lambda x : x*( 1-x )
lr = 2
momentum = 0.01


def predict(  ):
	pass


input_vector = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
target_vector = np.array( [ [0], [1], [1], [1] ] )

hidden_nodes = 2
hidden_weights = np.array( [ [ ( random()-0.5)*0.2 for x in range(0, hidden_nodes) ] for x in range( 0, input_vector.shape[1] ) ] )
output_weights = np.array( [ [ ( random()-0.5)*0.2  for x in range(0, 1) ] for x in range(0, hidden_nodes) ] )

print "\n\n hidden weights : ", hidden_weights, "\n output weights : ", output_weights

layer_1_error = None
prev_layer_1_error = None
layer_2_error = np.zeros( target_vector.shape )
prev_layer_2_error = np.zeros( target_vector.shape )
layer_1_updates = np.zeros( hidden_weights.shape )
layer_2_updates = np.zeros( output_weights.shape )

error_sum = 1

layer_1_input = input_vector
for i in range(0, 500):

	layer_1_output = sigmoid( np.dot( layer_1_input, hidden_weights ) )

	layer_2_input = layer_1_output
	layer_2_output = sigmoid( np.dot( layer_2_input, output_weights ) )

	prev_layer_1_error = layer_1_error
	prev_layer_2_error = layer_2_error

	layer_2_error = (target_vector-layer_2_output)*(1-layer_2_output)*layer_2_output
	layer_1_error = np.dot(layer_2_error, np.transpose(output_weights))*layer_1_output*(1-layer_1_output)

	layer_1_updates = lr*np.dot( np.transpose(layer_1_input), layer_1_error ) + momentum*layer_1_updates
	hidden_weights = hidden_weights+layer_1_updates

	layer_2_updates = lr*np.dot( np.transpose(layer_2_input), layer_2_error ) + momentum*layer_2_updates
	output_weights = output_weights+layer_2_updates


	error_sum = np.sum( square( target_vector-layer_2_output ) )
	if error_sum<0.005:
		print "interrupted" 
		break


print output_weights, "\n", 
print "\n\n hidden weights : ", hidden_weights, "\n output weights : ", output_weights, "\n output : ", layer_2_output