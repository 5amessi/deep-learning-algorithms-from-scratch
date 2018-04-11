import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate

    def fit(self, X, Y):
        self.weight = np.zeros(X.shape[1])
        self.bias = 0
        self.cost_ = []
        self.gradient_devent(X,Y)

    def gradient_devent(self,X,Y):
        for i in range(self.__epochs):
            y_ = np.dot(X, self.weight.T) + self.bias

            predicted_y = self.__sigmoid(y_)

            errors = (Y - predicted_y)

            self.weight += self.__learning_rate * errors.T.dot(X)

            self.bias += self.__learning_rate * np.sum(errors)

            self.cost_.append(self.__logit_cost(Y, predicted_y))

    def __logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.weight.T) + self.bias
        return np.where(self.__sigmoid(z) >= 0.5, 1, 0)

    def plot(self):
        plt.plot(range(1, len(self.cost_) + 1), np.log10(self.cost_))
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Logistic Regression - Learning rate 0.1')
        plt.tight_layout()
        plt.show()

def read_dataset(Normalize = 1):
    train = pd.read_csv('../Dataset/Titanic/train.csv')
    train_y = train['Survived']
    train['Sex'] = train['Sex'].replace(['male','female'] , (1,0))
    train['Embarked'] = train['Embarked'].replace(['C','S','Q'] , (1,2,3))
    train = train.drop(['PassengerId','Survived','Name','Ticket','Cabin'] , axis=1)
    train_x = np.asarray(train)
    train_y = np.asarray(train_y)
    train_x = np.nan_to_num(train_x)
    train_x, test_x , train_y,test_y = train_test_split(train_x, train_y,test_size=0.2, random_state=0)
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y

train_x ,train_y ,test_x ,test_y = read_dataset(Normalize=1)

lr = LogisticRegression(learning_rate=0.01, epochs=5000)

lr.fit(train_x ,train_y)

predict_y = lr.predict(test_x)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)
