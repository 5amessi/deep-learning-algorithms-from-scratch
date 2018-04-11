import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from helper_function import *

class DeepNN(object):
    def __init__(self, learning_rate=0.01, epochs=50,input_dim = 0):
        self.__epochs= epochs
        self.__learning_rate = learning_rate
        self.layers = []
        self.paramters = {}
        self.dparamters = {}
        self.layers.append({"un" : input_dim,"act" : 'None'})
        self.paramters['act' + str(0)] = self.layers[0]['act']

    def add_layer(self , units , activation):
        self.layers.append({"un" : units,"act" : activation})

    def inittialze(self):
        l = len(self.layers)
        if(l > 1):
            for i in range (1,l):
                self.paramters['W' + str(i)] = np.random.randn(self.layers[i-1]['un'],self.layers[i]['un'])*0.01
                self.paramters['b' + str(i)] = np.zeros((1,self.layers[i]['un']))
                self.paramters['act' + str(i)] = self.layers[i]['act']
        else:
            print("wrong num of layerss")


    def linear_activation_forward(self , A_prev, W, b, activation):
        Z = A_prev.dot(W) + b
        A_ = activation(activation,Z)
        return A_

    def feedforward(self,X):
        l = len(self.layers)
        A = X
        self.paramters['A' + str(0)] = A
        for i in range(1, l):
            A_prev = A
            W = self.paramters['W' + str(i)]
            b = self.paramters['b' + str(i)]
            act = self.paramters['act'+str(i)]
            A = self.linear_activation_forward(A_prev, W, b, act)
            self.paramters['A'+str(i)] = A

    def linear_activation_backward(self,A_prev, W ,dZ,dactivation):
        dA_prev = dactivation(dactivation,A_prev)

        dW = A_prev.T.dot(dZ)

        db = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dZ.dot(W.T)*(dA_prev)

        return dA_prev, dW, db

    def back_probagation(self,Y):
        l = len(self.layers)
        final_layer = self.paramters['A'+str(l-1)]
        final_layer_activation = self.paramters['act'+str(l-1)]
        error = Y - final_layer

        if (l > 2):
            dZ = error * dactivation(final_layer_activation, final_layer)
        else:
            dZ = error

        cost = np.sum(np.argmax(Y,axis=1)+1 != np.argmax(final_layer,axis=1)+1)/len(error)

        for i in reversed(range(1,l)):

            W = self.paramters['W' + str(i)]
            A_prev = self.paramters['A' + str(i-1)]
            A_prev = np.array(A_prev)
            W = np.array(W)
            A_prev_activation = self.paramters['act' + str(i-1)]

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(A_prev,W,dZ,A_prev_activation)

            self.dparamters["dW" + str(i)] = dW_temp
            self.dparamters["db" + str(i)] = db_temp

            dZ = np.copy(dA_prev_temp)

        return cost

    def fit(self, X, Y):
        Y = np.array(One_Hot(Y))
        self.cost_ = []
        self.inittialze()
        for i in range(self.__epochs):
            self.feedforward(X)
            cost = self.back_probagation(Y)
            self.update()
            self.cost_.append(cost)
            if i%100 == 0:
                print("epoch " ,i ,"loos = ", cost)
            #self.cost_.append(self.__cost(self._cross_entropy(output=activated_y, y_target=y)))

    def update(self):
        l = len(self.layers)
        for i in range(1, l):
            dw = self.dparamters["dW" + str(i)]
            db = self.dparamters["db" + str(i)]
            self.paramters['W' + str(i)] += self.__learning_rate * dw
            self.paramters['b' + str(i)] += self.__learning_rate * db

    def cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def cost(self, cross_entropy):
        return 0.5 * np.mean(cross_entropy)

    def predict(self, X):
        l = len(self.layers)
        A = X
        for i in range(1, l):
            A_prev = A
            W = self.paramters['W' + str(i)]
            b = self.paramters['b' + str(i)]
            act = self.paramters['act'+str(i)]
            A = self.linear_activation_forward(A_prev, W, b, act)
        max_indices = np.argmax(A,axis=1)+1
        return max_indices

    def plot(self):
        plt.plot(range(1, len(lr.cost_) + 1),(lr.cost_))
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Softmax Regression - Learning rate 0.02')
        plt.tight_layout()
        plt.show()

def read_dataset(Normalize = 1):
    train = pd.read_csv('Iris Data.txt')#read dataset
    train['Class'] = train['Class'].replace(["Iris-setosa","Iris-versicolor","Iris-virginica"] , (1,2,3))
    train_y = train['Class']
    train = train.drop(['Class'] , axis=1)
    train_x = np.asarray(train)
    train_y = np.asarray(train_y)
    train_x = np.nan_to_num(train_x)
    train_x, test_x , train_y,test_y = train_test_split(train_x, train_y,test_size=0.2, random_state=50)
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y

train_x ,train_y ,test_x ,test_y = read_dataset()

lr = DeepNN(learning_rate=0.01, epochs=1000,input_dim = 4)

lr.add_layer(1024 , 'tanh')
lr.add_layer(3 , 'sigmoid')

lr.fit(train_x, train_y)

lr.plot()

predicted_test = np.asarray(lr.predict(test_x))

print("expected y  = " ,test_y)

print("predicted y = " ,predicted_test)

correct = np.sum(predicted_test == test_y)

print ("%d out of %d predictions correct" % (correct, len(predicted_test)))

print("accuracy = ", correct/len(predicted_test)*100)
