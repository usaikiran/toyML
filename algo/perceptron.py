
import numpy as np
from random import random


epochs = 20
lr = 0.05
activate = np.vectorize( lambda x : 1 if x>0.5 else 0 )


class Perceptron:

    def __init__( self, inputSize, targetVector ):

        self.inputSize = inputSize
        self.targetVector = targetVector

        self.__generateWeights__()


    def __generateWeights__(self):

        self.weights = np.array( [ [ (random()-0.5)*2 for x in range(0, self.inputSize) ] ]*self.targetVector.shape[1] )


    def predict( self, inputVector ):

        Y = np.dot( inputVector, np.transpose(self.weights) )
        return activate(Y)


    def train( self, inputVector ):

        for e in range( epochs ):

            A = self.predict( inputVector )
            E = self.targetVector - A
            self.weights = self.weights + np.dot( np.transpose(E), inputVector )*lr


if __name__ == "__main__":

    I = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
    T = np.array( [ [1], [0], [0], [1] ] )

    perceptron = Perceptron( 2, T )
    perceptron.train( I )
    print perceptron.predict( I )
