import numpy as np


def __activate(outputs, threshold):
    return (outputs > threshold) * 1


def __derive_accuracy(outputs, target):
    return np.count_nonzero((target - outputs) == 0)/outputs.shape[0]


def train(inputs, target, threshold):
    weights = np.array([[0.5], [0.5]])
    learning_rate = 0.001

    max_iterations = 300
    iterations = 0
    accuracy = 0

    while accuracy < 0.8 and iterations < max_iterations:
        outputs = __activate(np.dot(inputs, weights), threshold)
        delta = np.subtract(target, outputs)
        aggregated_delta = np.dot(inputs.transpose(), delta) * learning_rate
        weights = weights + aggregated_delta
        iterations += 1
        accuracy = __derive_accuracy(outputs, target)

    print("\niterations: ", iterations, ", accuracy: ", accuracy)
    return weights


if __name__ == '__main__':
    inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    target = np.array([[1], [1], [1], [0]])
    threshold = 0.7

    weights = train(inputs, target, threshold)
    out = np.dot(inputs, weights)
    res = __activate(out, threshold)

    print("\nweights:", weights)
    print("\nout:", out)
    print("\nres:", res)
