# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:53:37 2018

@author: ZCC
"""


import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
df = pd.read_csv("zhengqi_train.txt", sep='\t')
test=pd.read_csv("zhengqi_test.txt", sep='\t')


#result= pd.read_csv("result.txt", sep='\t').values
y_target = np.array(df['target']).reshape((-1,1))
x_data = df.drop(['target'], axis=1)


e=["V3","V20","V27",'V17','V18','V22','V23','V24','V28','V35']
for key in x_data.keys():
    c=np.corrcoef(df[key].values, df.target.values)[0][1]
    print(key,": ",c)
    if( c<0.5 and c>-0.5 ):
        e.append(key) if key not in e else 0

for key in e:
    x_data = x_data.drop([key], axis=1)
    test=test.drop([key], axis=1)

datasize=x_data.shape[1]
print(datasize)
x_test=np.array(test.values).reshape([-1,datasize])
x_data =np.array(x_data).reshape([-1,datasize])

# 创建x，x是一个占位符（placeholder），代表待识别的图片
#
#
#
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) +  0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



def train(x_data,y_target,x_test):
    tmp = 10
    s=0
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, datasize])
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer
    l1 = add_layer(xs, datasize, 20, activation_function=tf.nn.sigmoid)
    # add output layer
    prediction = add_layer(l1, 20, 1, activation_function=None)

    # the error between prediction and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
    # important step
    sess = tf.Session()
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(150000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_target})
        if i % 1000 == 0:
            # to see the step improvement
            loss1=sess.run(loss, feed_dict={xs: x_data[2000:-1], ys: y_target[2000:-1]})
            print(i/1000,": ",loss1)

            if(loss1>tmp):
                s=s+1
            else:
                if (s>0):
                    s = s - 1
            if (loss1 < 0.15 and s>3):
                break
            tmp=loss1
    result=sess.run(prediction , feed_dict={xs: x_test})
    result2 = sess.run(prediction, feed_dict={xs: x_data})
    return result,result2
result,result2=train(x_data,y_target,x_test)

r= pd.DataFrame (result , columns = [" "])
r.to_csv ("result.txt" ,header=0,index=0)
print(result,result.shape)
plt.figure(1)
plt.plot(df.V0)
plt.plot(y_target)
plt.plot(result2)
plt.figure(2)
plt.plot(test.V0)
plt.plot(result)
plt.show()

# for i in range(38):
#     print("V",i,": ",cc[i][39])