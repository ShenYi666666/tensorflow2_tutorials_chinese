# -*- coding: utf-8 -*-
# @Time    : 2020/11/9 16:21
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

#1.1模型堆叠
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation=keras.activations.relu))
model.add(layers.Dense(32, activation=keras.activations.relu))
model.add(layers.Dense(10, activation=keras.activations.softmax))
#1.2网络配置
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.categorical_crossentropy, metrics=[tf.keras.metrics.categorical_accuracy])

#2.函数式API构建模型
input_x = tf.keras.Input(shape=(72, ))
hidden1 = layers.Dense(32, activation='relu')(input_x)
hidden2 = layers.Dense(16, activation='relu')(hidden1)
pred = layers.Dense(10, activation='softmax')(hidden2)
#构建tf.keras.Model实例
model = tf.keras.Model(input_x, outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit()

#3.模型子类化
class Mymodel(tf.keras.Model):
    def __init__(self, num_class=10):
        super(Mymodel, self).__init__(name='my_model')
        self.numclass = num_class
        #定义网络层
        self.layer1 = layers.Dense(32, activation='relu')
        self.layer2 = layers.Dense(num_class, activation='softmax')
    def __call__(self, inputs):
        #定义前向传播
        h1 = self.layer1(inputs)
        outputs = self.layer2(h1)
        return outputs

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.numclass
        return tf.TensorShape(shape)
#实例化模型类，并训练
model = Mymodel(num_class=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=10)

#自定义层
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel1', shape=shape, initializer='uniform', trainable=True)

        super(MyLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#使用自定义网络层构建模型
model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation(keras.activations.softmax)
])

model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=10)

