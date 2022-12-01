import numpy as np
import math
import matplotlib.pyplot as plt

class logistic:

    def __init__(self):
        self.weights = []
        self.weights0 = []

    def logistic(self, X, Y):
        w = np.zeros((X.shape[1], 1))
        w0 = 0
        alpha = 0.001
        cst = []
        it = []

        for i in range(100000):
            Z = (np.dot(w.T, X.T) + w0).T
            A = 1 / ( 1 + np.exp(-Z))
            A = A.T
            cost = -(1/X.shape[0]) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
            dw = (1/X.shape[0]) * np.dot((A - Y), X)
            dw0 = (1/X.shape[0]) * np.sum(A - Y)
            w = w - alpha * (dw).T
            w0 = w0 - alpha * (dw0)
            it.append(i)
            cst.append(cost)

        # plt.plot(it, cst)
        # plt.show()
        return w, w0

    def find_accuracy(self, y_pred, y_test):
        accuracy = 0
        n = y_test.shape[0]
        for i in range(0, n):
            if(y_test[i] == y_pred[i]):
                accuracy+=1
        return (accuracy / n) * 100


    def predict(self, x_test):
        w = self.weights
        w0 = self.weights0
        y_pred = []
        y_pred = np.dot(x_test, w.T) + w0.T
        y_pred = np.argmax(y_pred, axis = 1) + 1
        return y_pred


    def fit(self, X_train, y_train, no_of_classes):
        for i in range(no_of_classes):
            y_train_i = []
            for j in range(X_train.shape[0]):
                if(y_train[j] == i + 1):
                    y_train_i.append(1)
                else :
                    y_train_i.append(0)
            y_train_i = np.array(y_train_i)
            w, w0 = self.logistic(X_train, y_train_i)
            w = w.T
            self.weights.append(w[0])
            self.weights0.append(w0)

        self.weights = np.array(self.weights)
        self.weights0 = np.array(self.weights0)
        return self.weights, self.weights0