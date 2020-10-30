# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 10:33
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, '    ', y_train.shape)
print(x_test.shape, '     ', y_test.shape)

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

model = keras.Sequential()

#添加卷积层
model.add(layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), filters=32, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))

#池化层
model.add(layers.MaxPool2D(pool_size=(2, 2)))

#全连接层
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))

#分类层
model.add(layers.Dense(10, activation='softmax'))

#模型配置
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )
model.summary()

#模型训练
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['traning', 'validation'], loc='upper left')
plt.show()

result = model.evaluate(x_test, y_test)

