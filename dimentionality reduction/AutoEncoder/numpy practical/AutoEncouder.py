import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)

n_hidden_1 = 0##(128,265,512,1024)# 1st layer num featursses
n_input = 784  # MNIST data input (img shape: 28*28)

class auto(object):
    def __init__(self,learning_rate=0.01):
        self.lr = learning_rate
        self.w_xh = np.random.randn(n_input, n_hidden_1)
        self.w_hy = np.random.randn(n_hidden_1, n_input)
        self.b_h = np.random.randn(1, n_hidden_1)
        self.b_y = np.random.randn(1, n_input)

    def AutoEncouder(self,x , y):
        ##write your code here
        return;
    def activation_function(self, z):
        ##write your code here
        return;

    def activation_function_derivative(self,x):
        ##write your code here
        return;

    def predict(self, X):
        ##write your code here
        return;

epic = 0 #(0-20)
batch_size = 0 #(128-1024)
batches_number = 0 #(0-50)

##divide dataset into batches
total_batch = int(mnist.train.num_examples / batch_size) ##if you want to use all batches but it will be to slow on CPU
au = auto(learning_rate=0.01)
for i in range(epic):
    for k in range(batches_number):
        train_x,_ = mnist.train.next_batch(batch_size)
        au.AutoEncouder(train_x, train_x)
    print("Epoch:", '%04d' % (i+1))

###testing
examples_to_show = 10
display_step = 1
test = mnist.test.images[:examples_to_show]
encode_decode = au.predict(test)#####should implement it
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()