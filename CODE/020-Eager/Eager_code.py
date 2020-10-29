# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 19:41
# @Author  : ShenYi
import tensorflow as tf

#print(tf.add([3, 8], [2, 5]))
x = tf.matmul([[3], [6]], [[2]])
#print(x)
#print(tf.__version__)
layer = tf.keras.layers.Dense(100, input_shape= (None, 20))

#实现自定义网络层  扩展tf.keras.Layer类实现



#创建一个包含多个网络层的结构， 一般继承与tf.keras.Model類
class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name='resnet_block')

        #每个子层卷积核数
        filter1 , filter2 , filter3 = filters
        self.conv1 = tf.keras.layers.Conv2D(filter1, (1, 1) )
        self.bn1 = tf.keras.layers.BatchNormalization()

        #第二个子层
        self.conv2 = tf.keras.layers.Conv2D(filter2, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        #第三个子层
        self.conv3 = tf.keras.layers.Conv2D(filter3, (1, 1))
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, tranning=False):
        #堆叠每个子层
        x = self.conv1(inputs)
        x = self.bn1(x, tranning = tranning)

        x = self.conv2(x)
        x = self.bn2(x, tranning = tranning)

        x = self.conv3(x)
        x = self.bn3(x, tranning = tranning)

        #残差连接
        x += inputs
        outputs = tf.nn.relu(x)

        return outputs

resnetBlock = ResnetBlock(2, [6,4,9])
#数据测试
print(resnetBlock(tf.ones([1,3,9,9])))




