'''
- A simple multilayer perceptron with fixed no of hidden layers
'''

from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math

 	
sigmoid = np.vectorize( lambda x: 1/( 1+math.exp(-x) ) )
binary_activation = np.vectorize( lambda x: 0 if x<0.5 else 1 )
square = np.vectorize( lambda x: x*x )

derivative = lambda x : x*( 1-x )
lr = 0.9
momentum = 0


def predict(  ):
	pass


input_vector = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
target_vector = np.array( [ [0], [1], [1], [0] ] )

hidden_nodes_1 = 3
hidden_nodes_2 = 3
# hidden_weights_1 = np.array( [ [ ( random()-0.5)*0.2 for x in range(0, hidden_nodes_1) ] for x in range( 0, input_vector.shape[1] ) ] )
# hidden_weights_2 = np.array( [ [ ( random()-0.5)*0.2 for x in range(0, hidden_nodes_2) ] for x in range( 0, hidden_nodes_1 ) ] )
# output_weights = np.array( [ [ ( random()-0.5)*0.2 for x in range(0, 1) ] for x in range(0, hidden_nodes_2) ] )

hidden_weights_1 = np.array( [[-0.09929317, -0.08342229, -0.09614182],
 [-0.01904334,  0.02717919, -0.03740001]] )

hidden_weights_2 = np.array( [[-0.0782394, 0.04632704, -0.04953208],
 [ 0.05045396, -0.07656003, -0.09492723],
 [ 0.05774203,  0.09734058,  0.06919198]] )

output_weights = np.array( [[-0.03527433],
 [-0.05856922],
 [ 0.06033958]] )

print "\n\n hidden weights 1 : ", hidden_weights_1, "hidden weights 2 : ", hidden_weights_2, "\n output weights : ", output_weights

layer_1_error = None
prev_layer_1_error = None
layer_1_updates = np.zeros( hidden_weights_1.shape )

layer_2_error = None
prev_layer_2_error = None
layer_2_updates = np.zeros( hidden_weights_2.shape )

layer_3_error = None
prev_layer_3_error = np.zeros( target_vector.shape )
layer_3_updates = np.zeros( output_weights.shape )

bias = -1

error_sum = 1

layer_1_input = input_vector
for i in range(0, 20000):

	layer_1_output = sigmoid( np.dot( layer_1_input, hidden_weights_1 ) + bias )

	layer_2_input = layer_1_output
	layer_2_output = sigmoid( np.dot( layer_2_input, hidden_weights_2 ) + bias )

	layer_3_input = layer_2_output
	layer_3_output = sigmoid( np.dot( layer_3_input, output_weights ) + bias )

	prev_layer_1_error = layer_1_error
	prev_layer_2_error = layer_2_error
	prev_layer_3_error = layer_3_error


	layer_3_error = (target_vector-layer_3_output)*(1-layer_3_output)*layer_3_output
	layer_3_updates = lr*np.dot( np.transpose(layer_3_input), layer_3_error ) + momentum*layer_3_updates
	output_weights = output_weights+layer_3_updates

	layer_2_error = np.dot(layer_3_error, np.transpose(output_weights))*layer_2_output*(1-layer_2_output)
	layer_2_updates = lr*np.dot( np.transpose(layer_2_input), layer_2_error ) + momentum*layer_2_updates
	hidden_weights_2 = hidden_weights_2+layer_2_updates

	layer_1_error = np.dot(layer_2_error, np.transpose(hidden_weights_2))*layer_1_output*(1-layer_1_output)
	layer_1_updates = lr*np.dot( np.transpose(layer_1_input), layer_1_error ) + momentum*layer_1_updates
	hidden_weights_1 = hidden_weights_1+layer_1_updates

	#print "\n\n layer 1 updates : ", layer_1_updates, "\n layer 2 updates : ", layer_2_updates, "\n layer 3 updates : ", layer_3_updates

	error_sum = np.sum( square( target_vector-layer_3_output ) )
	if error_sum<0.001:
		print "interrupted at " + str(i)
		break


print "\n\n layer 1 input : ", layer_1_input, "\n layer 2 input : ", layer_2_input, "\n layer 3 input : ", layer_3_input
print "\n\n layer 1 error : ", layer_1_error, "\n layer 2 error : ", layer_2_error, "\n layer 3 error : ", layer_3_error
print "\n\n layer 1 weights : ", hidden_weights_1, "\n layer 2 weights : ", hidden_weights_2, "\n layer 3 weights : ", output_weights
print "\n\n output : ", layer_3_output