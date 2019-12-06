
"""
Created on Mon Jan  7 20:07:17 2019

@author: Zhou Cheng
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
import tensorflow.contrib.slim as slim
import h5py

filename = 'S:/Tianchi/data/data_shuffle.npy'
data = np.load(filename)
train = data[:50000, 8192:]
vali = data[50500:, 8192:]
print('The shape of train is ',train.shape)
print('The shape of vali is ',vali.shape)

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def bottelneck_layer(x,W1,W2):
    x = slim.batch_norm(x)
    x = tf.nn.relu(conv2d(x,W1))
    x = slim.batch_norm(x)
    x = tf.nn.relu(conv2d(x,W2))
    return x

def dense_block(input_x, W1_1,W1_2,W2_1,W2_2,W3_1,W3_2,W4_1,W4_2,W5_1,W5_2):
    layers_concat = []
    layers_concat.append(input_x)
    
    x = bottelneck_layer(input_x,W1_1,W1_2)
    layers_concat.append(x)
    
    x = tf.concat(layers_concat, axis = 3)
    x = bottelneck_layer(x,W2_1,W2_2)
    layers_concat.append(x)
    
    x = tf.concat(layers_concat, axis = 3)
    x = bottelneck_layer(x,W3_1,W3_2)
    layers_concat.append(x)
    
    x = tf.concat(layers_concat, axis = 3)
    x = bottelneck_layer(x,W4_1,W4_2)
    layers_concat.append(x)
    
    x = tf.concat(layers_concat, axis = 3)
    x = bottelneck_layer(x,W5_1,W5_2)
    
    return x


x = tf.placeholder("float", [None, 10240])
y_ = tf.placeholder("float", [None, 17])
keep_prob = tf.placeholder("float")
is_train = tf.placeholder("bool")
x_image = tf.reshape(x, [-1, 32, 32, 10])


db1_w1_1 = weight_variable([1,1,10,5])
db1_w1_2 = weight_variable([3,3,5,64])

db1_w2_1 = weight_variable([1,1,74,37])
db1_w2_2 = weight_variable([3,3,37,64])

db1_w3_1 = weight_variable([1,1,138,69])
db1_w3_2 = weight_variable([3,3,69,64])

db1_w4_1 = weight_variable([1,1,202,101])
db1_w4_2 = weight_variable([3,3,101,64])

db1_w5_1 = weight_variable([1,1,266,133])
db1_w5_2 = weight_variable([3,3,133,64])

result_db1 = dense_block(x_image,db1_w1_1,db1_w1_2,db1_w2_1,db1_w2_2,db1_w3_1,db1_w3_2,db1_w4_1,db1_w4_2,db1_w5_1,db1_w5_2)


#w_conv1_1 = weight_variable([3, 3, 10, 64])
#b_conv1_1 = bias_variable([64])
#h_conv1_1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(x_image, w_conv1_1) + b_conv1_1,is_training = is_train,decay = 0.9))
h_pool1 = max_pool_2x2(result_db1)

w_conv2_1 = weight_variable([3, 3, 64, 128])
b_conv2_1 = bias_variable([128])
w_conv2_2 = weight_variable([3, 3, 128, 128])
b_conv2_2 = bias_variable([128])
h_conv2_1 = tf.nn.relu(conv2d(h_pool1, w_conv2_1) + b_conv2_1)
h_conv2_2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h_conv2_1, w_conv2_2) + b_conv2_2,is_training = is_train,decay = 0.9))
h_pool2 = max_pool_2x2(h_conv2_2)

w_conv3_1 = weight_variable([3, 3, 128, 256])
b_conv3_1 = bias_variable([256])
w_conv3_2 = weight_variable([3, 3, 256, 256])
b_conv3_2 = bias_variable([256])
w_conv3_3 = weight_variable([3, 3, 256, 256])
b_conv3_3 = bias_variable([256])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2, w_conv3_1) + b_conv3_1)
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, w_conv3_2) + b_conv3_2)
h_conv3_3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h_conv3_2, w_conv3_3) + b_conv3_3,is_training = is_train,decay = 0.9))
h_pool3 = max_pool_2x2(h_conv3_3)

h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*256])
w_fc1 = weight_variable([4*4*256, 512])
b_fc1 = bias_variable([512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = weight_variable([512, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
w_fc3 = weight_variable([512, 17])
b_fc3 = bias_variable([17])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

gamma = 1e-2
delta = 1e-7
learning_rate = 1e-4
#L2_norm = gamma*(tf.norm(w_fc3)+tf.norm(w_fc2)+tf.norm(w_fc1)+
               #  +tf.norm(w_conv2_1)+tf.norm(w_conv2_2)+tf.norm(w_conv3_1)+tf.norm(w_conv3_2)+tf.norm(w_conv3_3))
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv+delta))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

def get_batch(data, batch_size):
    sample = random.sample(list(data),batch_size)
    sample = np.array(sample)
    train_x = sample[:,:-17]
    train_y = sample[:,-17:]
    return train_x, train_y
	
Loss = []
vali_Eval = []
train_Eval = []
Loss_vali = []
batch_size = 256
step_num = 12000
for i in range(step_num):
    batch_x,batch_y = get_batch(train,batch_size)
    train_step.run(feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5, is_train : True})
    if i%200 == 0:
        temp_loss = cross_entropy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob:1.0, is_train : False})
        vali_accuracy = accuracy.eval(feed_dict={x:vali[:,:-17], y_:vali[:,-17:], keep_prob:1.0, is_train : False})
        train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob:1.0, is_train : False})
        Loss.append(temp_loss)
        vali_Eval.append(vali_accuracy)
        train_Eval.append(train_accuracy)
        train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob:1.0, is_train : False})
        vali_loss = cross_entropy.eval(feed_dict={x:vali[:,:-17], y_:vali[:,-17:], keep_prob:1.0, is_train : False})
        Loss_vali.append(vali_loss)
        print ("step %d, train accuracy %g, vali accuracy %g, train loss %g,vali loss %g" %(i, train_accuracy, vali_accuracy, temp_loss,vali_loss))

pred = y_conv.eval(feed_dict={x:vali[:,:-17], y_:vali[:,-17:], keep_prob:1.0,is_train : False})
train_val_y = np.argmax(vali[:,-17:],axis = 1)
pred_y = np.argmax(pred, axis = 1)
print (classification_report(train_val_y, pred_y))

filename = 'S:/Tianchi/data/round1_test_b_20190104.h5'
f = h5py.File(filename,'r')
test_s2 = f['sen2']
test = []
for i in range(0,test_s2.shape[0]):
    temp2 = test_s2[i].flatten()
    test.append(temp2)
test = np.array(test)
test_y = np.zeros((test.shape[0],17))
pred = tf.argmax(y_conv, 1)
test_x_0 = test[0:1500]
test_y_0 = test_y[0:1500]
P_0 = pred.eval(feed_dict={x:test_x_0, y_:test_y_0, keep_prob:1.0,is_train : False})
test_x_1 = test[1500:3000]
test_y_1 = test_y[1500:3000]
P_1 = pred.eval(feed_dict={x:test_x_1, y_:test_y_1, keep_prob:1.0,is_train : False})
test_x_2 = test[3000:4500]
test_y_2 = test_y[3000:4500]
P_2 = pred.eval(feed_dict={x:test_x_2, y_:test_y_2, keep_prob:1.0,is_train : False})
test_x_3 = test[4500:]
test_y_3 = test_y[4500:]
P_3 = pred.eval(feed_dict={x:test_x_3, y_:test_y_3, keep_prob:1.0,is_train : False})
P = np.hstack([P_0,P_1,P_2,P_3])
one_hot=tf.one_hot(P,17)
Pred_one_hot = sess.run(one_hot)
Pred_one_hot = Pred_one_hot.astype(np.int32)
out = pd.DataFrame(Pred_one_hot, columns = list(range(17)))
print(out.head())
outname = 'ceshi4BN2222.csv'
out.to_csv(outname, index = False, header = False)