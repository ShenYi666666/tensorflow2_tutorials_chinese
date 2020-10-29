# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 15:19
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=7)
print(x_train.shape, '   ', y_train.shape)
print(x_test.shape, '    ', y_test.shape)

#构建模型
model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(30,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1)

model.evaluate(x_test, y_test)

print(model.metrics_names)