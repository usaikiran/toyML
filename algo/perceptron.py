
import numpy as np
from random import random


epochs = 20
lr = 0.05
activate = np.vectorize( lambda x : 1 if x>0.5 else 0 )


def predict( I, W ):

    Y = np.dot( I, np.transpose(W) )
    return activate(Y)


def train( I, W, T ):

    for e in range( epochs ):

        A = predict( I, W )
        E = T - A
        W = W + np.dot( np.transpose(E), I )*lr

    return W


if __name__ == "__main__":

    I = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
    T = np.array( [ [0], [0], [0], [1] ] )
    W = np.array( [ [ (random()-0.5)*2 for x in range(0, I.shape[1]) ] ]*T.shape[1] )

    W = train( I, W, T )
    print "WEIGHTS : ", W
    print predict( [ [0, 0], [0, 1], [1, 0], [1, 1] ], W )
