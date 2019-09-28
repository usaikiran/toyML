
from __future__ import division

from itertools import izip as zip
import numpy as np
from random import random
import math


SIGMA = 0.5

class RBF:

    def __init__( self, nRBF, targetVector ):

        self.nRBF = nRBF
        self.targetVector = targetVector


    def _calcDistance( self, inputVector, weights, nRBF ):

        out = np.zeros( ( inputVector.shape[0], weights.shape[1] ) )
        for i in range(nRBF):
            out[:,i] = np.exp( -np.sum( ( inputVector-np.ones( ( 1, inputVector.shape[1] ) )*weights[:,i] )**2, axis=1 )/( 2*(SIGMA**2) ) )

        return out


    def train( self, inputVector ):

        self.inputVector = inputVector
        indices = np.random.shuffle(range(inputVector.shape[0]))

        self.weights1 = np.zeros( (self.inputVector.shape[1], self.nRBF) )
        for i in range(self.nRBF):
            self.weights1[:,i] = self.inputVector[i,:]

        self.hidden = self._calcDistance( self.inputVector, self.weights1, self.nRBF )
        self.weights2 = np.dot( np.linalg.pinv(self.hidden), self.targetVector )


    def predict( self, inputVector ):

        hiddenOutput = self._calcDistance( inputVector, self.weights1, self.nRBF )
        out = np.dot( hiddenOutput, self.weights2 )

        return out


if __name__ == "__main__":
    
    I = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
    T = np.array( [ [1], [0], [1], [0] ] )

    rbf = RBF(3, T)
    rbf.train(I)
    print rbf.predict(I)