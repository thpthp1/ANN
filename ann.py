import numpy as np
from numpy.core.numeric import zeros_like
from numba import njit

@njit
def sigmoid(s): return 1.0 / (1.0 + np.exp(-s))

def sigmoid_grad(s): return sigmoid(s) * (1 - sigmoid(s))

@njit
def back_propagate(weights, bias, layers, y, outputs, alpha=0.1):
        errs = [np.zeros_like(b) for b in bias]
        proba = outputs[-1]
        errs[-1] = proba* (1.0 - proba)* (y - proba)
        for i in range(len(weights) - 2, -1, -1):
            errs[i] = outputs[i] * (1.0 - outputs[i]) * (errs[i + 1]@weights[i+1].T)
        for i in range(len(weights) - 1, -1, -1):
            weights[i] += alpha * outputs[i - 1].T * errs[i]
            bias[i] += alpha * errs[i]

@njit
def feed_forward(weights, bias, x):
        outputs = [np.zeros_like(b) for b in bias]
        output = x.reshape((1, -1))
        for i in range(len(weights)):
            #print(output, weights[i])
            output = sigmoid(np.dot(output, weights[i]) + bias[i])
            outputs[i] = output
        return outputs

@njit
def fit(weights, bias, layers, X, y, epoch=1000, alpha=0.1):
    for _ in range(epoch):
        for idx in range(X.shape[0]):
            #print(X[idx, :].reshape((1, -1)))
            outputs = feed_forward(weights, bias, X[idx, :])
            #print(outputs)
            back_propagate(weights, bias, layers, y[idx, :], outputs, alpha=alpha)
            #print(self.weights)

class ANNClassifier:

    def __init__(self, layers: list, init_val_gen='random') -> None:
        self.layers = layers
        # list of weight matrices
        self.weights = [None] * (len(layers) - 1)
        # list of biases
        self.bias = [None] * (len(layers) - 1)
        gen_func = self._get_gen_func(init_val_gen)
        for i in range(1, len(layers)):
            
            self.weights[i - 1] = gen_func((layers[i-1], layers[i]))
            self.bias[i - 1] = gen_func((1, layers[i]))

    @classmethod
    def _get_gen_func(cls, init_val_gen='random'):
        if init_val_gen == 'random':
            return lambda x: np.random.uniform(low=-1, high=1, size=x)
        elif init_val_gen == 'zero':
            return lambda x: np.zeros(size=x)

            
    def train(self, X: np.matrix, y, epoch=1000, alpha=0.1):
        '''
        For each epoch
            for each data sample:
                feed_forward, get proba
                calculate error
                back_prop
        '''
        # for _ in range(epoch):
        #     for idx in range(X.shape[0]):
        #         #print(X[idx, :].reshape((1, -1)))
        #         outputs = self._feed_forward(X[idx, :].reshape((1, -1)))
        #         self._back_propagate(y[idx, :], outputs, alpha=alpha)
        #         #print(self.weights)
        fit(self.weights, self.bias, self.layers, X, y, epoch, alpha)
    
    def predict_proba(self, X: np.ndarray):
        probas = np.zeros((X.shape[0], self.bias[-1].size))
        for idx in range(X.shape[0]):
            vec = X[idx, :]
            outputs = self._feed_forward(vec)
            probas[idx, :] = outputs[-1]
        return probas
            
    def _feed_forward(self, x):
        return feed_forward(self.weights, self.bias, x)

    def _back_propagate(self, y, outputs, alpha=0.1):
        # errs = [None] * (len(self.layers) - 1)
        # proba = outputs[-1]
        # errs[-1] = proba* (1 - proba)* (y - proba)
        # for i in range(len(self.weights) - 2, -1, -1):
        #     errs[i] = outputs[i] * (1.0 - outputs[i]) * (errs[i + 1]@self.weights[i+1].T)
        # for i in range(len(self.weights) - 1, -1, -1):
        #     self.weights[i] += alpha * outputs[i - 1].T * errs[i]
        #     self.bias[i] += alpha * errs[i]
        back_propagate(self.weights, self.bias, self.layers, y, outputs, alpha)
    def predict(self, X):
        if self.layers[-1] > 1:
            return np.apply_along_axis(np.argmax, 1, self.predict_proba(X))
        else: return self.predict_proba(X)

if __name__ == "__main__":
    model = ANNClassifier([2, 3, 2])
    weights = [
        np.array([
            [0.1, 0, 0.3],
            [-0.2, 0.2, -0.4]
        ]),
        np.array([
            [-0.4, 0.2],
            [0.1, -0.1],
            [0.6, -0.2]
        ])
    ]
    bias = [
        np.array(
            [[0.1, 0.2, 0.5]]
        ),
        np.array(
            [[-0.1, 0.6]]
        )
    ]
    model.weights = weights
    model.bias = bias
    model.train(np.array([[0.6, 0.1], [0.2, 0.3]]), np.array([[1, 0], [0, 1]]), epoch=1)

    ## Test on XOR
    model = ANNClassifier([2, 5, 1])
    model.train(np.array([
                            [1.0, 0.0], 
                            [0.0, 1.0],
                            [1.0, 1.0],
                            [0.0, 0.0]]), 
                np.array([
                            [1.0], 
                            [1.0],
                            [0.0],
                            [0.0]]), epoch=100000, alpha=1)
    print(model.weights)
    print(model.bias)
    print(model.predict_proba(np.array([
                            [1.0, 0.0], 
                            [0.0, 1.0],
                            [1.0, 1.0],
                            [0.0, 0.0]])))
    
    # Verify with in class example
    model = ANNClassifier([3, 2, 1])
    weights = [
        np.array([
            [0.2, -0.3],
            [0.4, 0.1],
            [-0.5, 0.2]
        ]),
        np.array([
            [-0.3],
            [-0.2]
        ])
    ]
    bias = [
        np.array(
            [[-0.4, 0.2]]
        ),
        np.array(
            [[0.1]]
        )
    ]
    model.weights = weights
    model.bias = bias
    model.train(np.array([[1.0, 0.0, 1.0]]), np.array([[1]]), epoch=1,alpha=0.9)
    # print(model.weights)
    # print(model.bias)