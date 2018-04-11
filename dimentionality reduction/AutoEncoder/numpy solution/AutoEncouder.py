import numpy as np
import matplotlib.pyplot as plt
from sympy.functions.elementary.complexes import principal_branch
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)
epic = 5
batch_size = 256
examples_to_show = 10
display_step = 1

n_hidden_1 = 512  # 1st layer num featursses
n_input = 784  # MNIST data input (img shape: 28*28)
class auto(object):
    def __init__(self,learning_rate=0.01):
        self.lr = learning_rate
        self.w_xh = np.random.randn(n_input, n_hidden_1)
        self.w_hy = np.random.randn(n_hidden_1, n_input)
        self.b_h = np.random.randn(1, n_hidden_1)
        self.b_y = np.random.randn(1, n_input)

    def AutoEncouder(self,x , y):
        for ind in range(x.shape[0]):
            #feeed forward
            input_x = np.reshape(x[ind] , (1,n_input))
            target_y = np.reshape(y[ind] , (1,n_input))
            h_lay = self.sigmoid(np.dot(input_x , self.w_xh)+self.b_h)
            output = self.sigmoid(np.dot(h_lay , self.w_hy)+ self.b_y)
            errors = target_y - output
            er = (output - target_y)**2
            cost = np.sqrt(np.sum(er) / errors.shape[1])
            #back propagation
            prop_out_lay = errors * self.dsigmoid(output)
            prop_hid_lay = np.dot(prop_out_lay , self.w_hy.T) * self.dsigmoid(h_lay)
            self.w_xh += self.lr * np.dot(input_x.T , prop_hid_lay)
            self.w_hy += self.lr * np.dot(prop_hid_lay.T , prop_out_lay)
            self.b_h += self.lr * prop_hid_lay
            self.b_y += self.lr * prop_out_lay
            #np.clip(self.w_xh, -5, 5, out=self.w_xh)
            #np.clip(self.w_hy, -5, 5, out=self.w_hy)
            np.clip(h_lay, 0.01, 555, h_lay)
            #np.clip(prop_hid_lay, 0.01, 555, prop_hid_lay)
            #np.clip(self.b_h, -5, 5, self.b_h)
            #np.clip(self.b_y, -5, 5, self.b_y)
        print(cost)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def dsigmoid(self,x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def predict(self, X):
        z = np.dot(X , self.w_xh)+self.b_h
        z = self.sigmoid(z)
        z2 = np.dot(z , self.w_hy)+self.b_y
        z2 = self.sigmoid(z2)
        return z2

"""""
au = auto(learning_rate=0.01)
train_x,_ = mnist.train.next_batch(5000)
for i in range(epic):
    au.AutoEncouder(train_x, train_x)
    print("Epoch:", '%04d' % (i+1))
"""

total_batch = int(mnist.train.num_examples / batch_size)
au = auto(learning_rate=0.001)
for i in range(epic):
    for k in range(50):
        train_x,_ = mnist.train.next_batch(batch_size)
        au.AutoEncouder(train_x, train_x)
    print("Epoch:", '%04d' % (i+1))



test = mnist.test.images[:examples_to_show]

encode_decode = au.predict(test)
#print (encode_decode)
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
