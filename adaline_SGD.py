import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Adaline_SGD:
    def __init__(self, eta=0.01,n_iter=10,shuffle=True,random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle=shuffle
        self.w_initialized = False
        self.random_state=random_state

    def fit(self,X,y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            losses = []
            for xi, target in zip(X,y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
            """Fit training data without reinitializing the weights"""
            if not self.w_initialized:
                self._initialize_weights(X.shape[1])
            if y.ravel().shape[0] > 1:
                for xi, target in zip(X, y):
                    self._update_weights(xi, target)
            else:
                self._update_weights(X, y)
            return self

    def _shuffle(self, X, y):
            """Shuffle training data"""
            r = self.rgen.permutation(len(y))
            return X[r], y[r]

    def _initialize_weights(self, m):
            """Initialize weights to small random numbers"""
            self.rgen = np.random.RandomState(self.random_state)
            self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
            self.b_ = np.float_(0.)
            self.w_initialized = True

    def _update_weights(self, xi, target):
            """Apply Adaline learning rule to update the weights"""
            output = self.activation(self.net_input(xi))
            error = (target - output)
            self.w_ += self.eta * 2.0 * xi * (error)
            self.b_ += self.eta * 2.0 * error
            loss = error ** 2
            return loss

    def net_input(self, X):
            """Calculate net input"""
            return np.dot(X, self.w_) + self.b_

    def activation(self, X):
            """Compute linear activation"""
            return X

    def predict(self, X):
            """Return class label after unit step"""
            return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 header=None, encoding='utf-8' )
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0,2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_sgd = Adaline_SGD(n_iter=15, eta=0.01, random_state=1)

ada_sgd.fit(X_std, y)
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')

plt.show()

ada_sgd.partial_fit(X_std[0, :], y[0])