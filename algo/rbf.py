
from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math

from mulltilayer_perceptron import MultilayerPerceptron


SIGMA = 0.5

class RBF:

    def __init__( self, nRBF, targetVector ):

        self.nRBF = nRBF
        self.targetVector = targetVector


    def _calcDistance( inputVector, weights, nRBF ):

        out = np.zeros( ( inputVector.shape[0], weights.shape[1] ) )
        for i in range(nRBF):
            out[:,i] = self.exp( -np.sum( math.pow( self.inputVector-np.ones((1,inputVector.shape[0])*self.weights1), 2 ) )/( 2*(SIGMA**2) ) )

        return out


    def train( self, inputVector ):

        self.inputVector = inputVector
        indices = np.random.shuffle(range(inputVector.shape))

        for i in range(nRBF):
            self.weights1[:,i] = self.inputVector[i,:]

        self.hidden = _calcDistance( inputVector, weights, nRBF )
        self.weights2 = np.dot( np.linalg.pinv(self.hidden), self.targetVector )


    def predict( self, inputVector ):

        hiddenOutput = _calcDistance( inputVector, weights, nRBF )
        out = np.dot( hiddenOutput, self.weights2 )

        return out


if __name__ == "__main__":
    
    