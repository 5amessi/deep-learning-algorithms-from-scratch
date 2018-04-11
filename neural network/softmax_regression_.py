import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class SoftmaxRegression(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate
        self.OneHotEn = {}

    def fit(self, X, y):
        self.w_ = np.zeros((X.shape[1], len(classes)))
        self.b = np.ones((1,len(classes)))
        self.cost_ = []

        for i in range(self.__epochs):
            y_ = self.__net_input(X, self.w_, self.b)
            activated_y = self.__activation(y_)
            errors = (y - activated_y)
            self.w_ += self.__learning_rate * X.T.dot(errors)
            self.b += self.__learning_rate * errors.sum()
            self.cost_.append(self.__cost(self._cross_entropy(output=activated_y, y_target=y)))

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def __cost(self, cross_entropy):
        return 0.5 * np.mean(cross_entropy)

    def __softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def __net_input(self, X, W, b):
        return (X.dot(W) + b)

    def __activation(self, X):
        return self.__softmax(X)

    def predict(self, X):
        z = self.__net_input(X, self.w_, self.b)
        activated_z = self.__softmax(z)
        max_indices = np.argmax(activated_z,axis=1)+1
        return max_indices

    def One_Hot(self,input):
        data = list(set(input))
        arr = np.zeros((len(data),len(data)))
        for i in range (len(data)):
            arr[i][i] = 1
            self.OneHotEn[data[i]] = arr[i]

    def encoding(self,train_y):
        lables = []
        for i in range(len(train_y)):
            lables.append(self.OneHotEn[train_y[i]])
        return lables

    def plot(self):
        plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Softmax Regression - Learning rate 0.02')
        plt.tight_layout()
        plt.show()

def read_dataset(Normalize = 1):
    train = pd.read_csv('../Dataset/Iris/Iris.csv')
    train['Species'] = train['Species'].replace(["Iris-setosa","Iris-versicolor","Iris-virginica"] , (1,2,3))
    train_y = train['Species']
    train = train.drop(['Species','Id'] , axis=1)
    train_x = np.asarray(train)
    train_y = np.asarray(train_y)
    train_x = np.nan_to_num(train_x)
    train_x, test_x , train_y,test_y = train_test_split(train_x, train_y,test_size=0.1, random_state=50)
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y

train_x ,train_y ,test_x ,test_y = read_dataset()

input_one_hot, classes = np.unique(train_y, return_counts=True)

lr = SoftmaxRegression(learning_rate=0.001, epochs=10000)

lr.One_Hot(train_y)

train_y = lr.encoding(train_y)

lr.fit(train_x, train_y)

predicted_test = np.asarray(lr.predict(test_x))

print("expected y  = " ,test_y)

print("predicted y = " ,predicted_test)

correct = np.sum(predicted_test == test_y)

print ("%d out of %d predictions correct" % (correct, len(predicted_test)))

print("accuracy = ", correct/len(predicted_test)*100)