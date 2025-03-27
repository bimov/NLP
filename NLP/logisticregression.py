import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.05, iterations = 500):
        self.lr = lr
        self.iterations = iterations
        self.w = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, positive, negative, w):
        dword = np.array([0.0] * len(w))
        for word in positive.keys():
            c_positive = (self._sigmoid(np.dot(positive[word], w)) - 1) * w
            dword += (self._sigmoid(np.dot(positive[word], w)) - 1) * positive[word]
            positive[word] -= c_positive * self.lr

        for word in negative.keys():
            c_negative = (self._sigmoid(np.dot(negative[word], w))) * w
            dword += (self._sigmoid(np.dot(negative[word], w))) * negative[word]
            negative[word] -= c_negative * self.lr
        w -= self.lr * dword
        return positive, negative, w


    def logistic_regression(self, positive, negative, w):
        new_pos, new_neg, new_w = 0, 0, 0
        for i in range(self.iterations):
            new_pos, new_neg, new_w = self.gradient(positive, negative, w)
            if np.abs(np.linalg.norm(w - new_w)) < 0.01:
                break
        return new_pos, new_neg, new_w





