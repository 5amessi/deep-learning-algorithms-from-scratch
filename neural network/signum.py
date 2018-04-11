import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class perceptron(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs = epochs
        self.__learning_rate = learning_rate

    def fit(self, X, Y):
        self.weight = np.zeros(X.shape[1])
        print(np.shape(X))
        self.bias = 0
        self.cost_ = []
        for i in range(self.__epochs):
            net_out = np.dot(X, self.weight.T) + self.bias

            predicted_y = self.signum(net_out)

            errors = (Y - predicted_y)

            self.weight += self.__learning_rate * errors.T.dot(X)
            if addbias == True:
                self.bias += self.__learning_rate * np.sum(errors)

    def signum(self, z):
        return np.where(z > 0, 1, -1)

    def predict(self, X):
        z = np.dot(X, self.weight.T) + self.bias
        return np.where(z > 0, 1, -1)

    def plot(self, X, y):
        print(np.shape(X[y == 1][:, 0]))
        print(np.shape(X[y == -1][:, 0]))
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', c='red')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='Class 2', c='blue')
        xmin = np.min(X[y == 1][:, 0], axis=0) - 1
        xmax = np.max(X[y == 1][:, 0], axis=0) + 1
        y1 = (-self.bias - self.weight[0] * xmin) / self.weight[1]
        y2 = (-self.bias - self.weight[0] * xmax) / self.weight[1]
        plt.plot([xmin, xmax], [y1, y2], 'b')
        plt.show()


def read_dataset(class1, class2, feature1, feature2):
    data = pd.read_csv('Iris Data.txt')  # read dataset
    data['Class'] = data['Class'].replace(["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
                                          (1, 2, 3))  # replace each name with num
    y = data['Class']  # Output
    x = data[[feature1, feature2]]  # input
    x = np.asarray(x)  # convert datatybe from DataFrame to array
    y = np.asarray(y)  # convert datatybe from DataFrame to array
    x = np.concatenate((x[y == class1], x[y == class2]))
    y = np.concatenate((y[y == class1], y[y == class2]))
    y[y == class1] = 1  # make class1 == 1
    y[y == class2] = -1  # make class2 == -1
    x = (x - x.min()) / (x.max() - x.min())  # normalize
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    return x, y, train_x, train_y, test_x, test_y


#################################################################################
learning_rate = 0.01
epochs = 50
class1 = 1
class2 = 2
feature1 = "X1"
feature2 = "X2"
addbias = True
x, y, train_x, train_y, test_x, test_y = read_dataset(class1, class2, feature1, feature2)  # read_dataset

sig = perceptron(learning_rate, epochs)  # load model

sig.fit(train_x, train_y)  # train model

predict_y = sig.predict(test_x)  # test model

correct = np.sum(predict_y == test_y)  # accuracy

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)
#####################################################################################3
sig.plot(train_x, train_y)
