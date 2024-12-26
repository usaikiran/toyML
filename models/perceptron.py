import random

import numpy as np


class Perceptron:

    def __init__(self, inputs, targets):
        self.epochs = 50
        self.learning_rate = 0.001
        self.weights = None
        self.activation_threshold = 0.5
        self.inputs = inputs
        self.targets = targets

    def __activate(self, outputs):
        return (outputs > self.activation_threshold) * 1

    def __generate_weights(self):
        return [[random.random() for _ in range(0, self.targets.shape[1])]] * self.inputs.shape[1]

    def predict(self):
        return self.__activate(np.dot(self.inputs, self.weights))

    def train(self):
        self.weights = self.__generate_weights()
        for i in range(self.epochs):
            errors = np.subtract(self.targets, self.predict())
            delta = np.dot(self.inputs.transpose(), errors) * self.learning_rate
            self.weights = self.weights + delta


if __name__ == '__main__':
    inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    targets = np.array([[1], [1], [1], [0]])

    model = Perceptron(inputs, targets)
    model.train()
    print(model.predict())
