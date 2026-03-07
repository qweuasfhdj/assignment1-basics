import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    '''
    sigmoid(x) (1 - sigmoid(x))
    :param x:
    :return:
    '''
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


class MLP:
    def __init__(self, sizes, lr, activation: 'relu'):
        self.sizes = sizes
        self.lr = lr
        self.layer_size = len(sizes)
        self.W = [np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2 / sizes[i]) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(1, sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, x):
        self.A = [x]
        self.Z = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.A[-1] @ W + b
            self.Z.append(z)
            if i == self.layer_size - 1:
                a = sigmoid(z)
            else:
                a = relu(z)
            self.A.append(a)

        return self.A[-1]

    def backward(self, label):
        num_samples = len(label)
        error = self.A[-1] - y
