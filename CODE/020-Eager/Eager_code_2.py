# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 11:38
# @Author  : ShenYi

import tensorflow as tf
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self):
        #初始化变量
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    def __call__(self, x):
        return self.W * x + self.b
model = Model()
print(model(2))

def loss(predict_y , true_y):
    return tf.reduce_mean(tf.square(predict_y- true_y))


TRUE_w = 3.0
TRUE_b = 2.0
num = 1000
#随机输入
input = tf.random.normal(shape=[num])
#随机噪声
noise = tf.random.normal(shape=[num])
#构造数据
output = TRUE_w * input + noise + TRUE_b

plt.scatter(input, output, c='b')
plt.scatter(input, model(input), c='r')
plt.show()

#当前loss
print(loss(model(input), output))


