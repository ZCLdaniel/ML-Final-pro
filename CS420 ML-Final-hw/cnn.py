import numpy as np
from PIL import Image
import tensorflow as tf

data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

data = np.fromfile("mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train_label",dtype=np.uint8)

testdata = np.fromfile("../mnist_test/mnist_test_data",dtype=np.uint8)
testlabel = np.fromfile("../mnist_test/mnist_test_label",dtype=np.uint8) 

#reshape the matrix
data = data.reshape(data_num,2025)
testdata = testdata.reshape(10000,2025)

#get the data and label combine
newdata = []
for i in range(60000):
    label1 = []
    newdata2 = []
    tem = int(label[i])
    if tem == 0:
        label1.append(1)
        for i in range(9):
            label1.append(0)
    elif tem !=9:
        for i in range(tem):
            label1.append(0)
        label1.append(1)
        for i in range(9-tem):
            label1.append(0)
    else:
        for i in range(9):
            label1.append(0)
        label1.append(1)
    label1 = np.array(label1)
    newdata2.append(data[i])
    newdata2.append(label1)
    newdata2 = np.array(newdata2)
    newdata.append(newdata2)
newdata = np.array(newdata)

#get the test.label come to 10_conponent
labelcov = []
for i in range(10000):
    label1 = []
    tem = int(testlabel[i])
    if tem == 0:
        label1.append(1)
        for i in range(9):
            label1.append(0)
    elif tem !=9:
        for i in range(tem):
            label1.append(0)
        label1.append(1)
        for i in range(9-tem):
            label1.append(0)
    else:
        for i in range(9):
            label1.append(0)
        label1.append(1)
    label1 = np.array(label1)
    labelcov.append(label1)
labelcov = np.array(labelcov)

#get my cnn moudel
x = tf.placeholder("float", shape=[None, 2025])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#the first cl
W_conv1 = weight_variable([6, 6, 1 , 32 ])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,45,45,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#pooling
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#the second cl
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#pooling
W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#sess.run()
sess = tf.InteractiveSession()
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv+1e-9))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#test
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

import sys
import random
print("Begin Training")
for i in range(20000): 
    slice = random.sample(newdata, 50)
    batch_x = []
    batch_y = []
    print i
    for j in range(50):
        batch_x.append(slice[j][0])
        batch_y.append(slice[j][1])
    batch_xs = np.array(batch_x)
    batch_ys = np.array(batch_y)
    if i%50== 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

#get the test accuracy
print "test accuracy %g"%accuracy.eval(feed_dict={x: testdata, y_: labelcov, keep_prob: 1.0})