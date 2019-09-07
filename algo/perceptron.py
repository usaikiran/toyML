
import numpy as np
from random import random


EPOCHS = 20
LR = 0.05
BIAS = -0.5

activate = np.vectorize( lambda x : 1 if x>0.5 else 0 )


class Perceptron:

    def __init__( self, inpDim, targetVector ):

        self.inpDim = inpDim
        self.targetVector = targetVector

        __generateWeights__()


    def __generateWeights__():

        self.weights = np.array( [ [ (random()-0.5)*2 for x in range(0, self.inpDim) ] ]*self.targetVector.shape[1] )


    def predict( self, inputVector ):

        Y = np.dot( inputVector, np.transpose(self.weights) )
        return activate(Y+BIAS)


    def train( self, inputVector ):

        for e in range( EPOCHS ):

            A = self.predict( inputVector )
            E = self.targetVector - A
            self.weights = self.weights + np.dot( np.transpose(E), inputVector )*LR


if __name__ == "__main__":

    I = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
    T = np.array( [ [1], [1], [1], [0] ] )

    perceptron = Perceptron( 2, T )
    perceptron.train( I )
    print perceptron.predict( I )
