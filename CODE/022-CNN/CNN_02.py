# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 14:05
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

num_features = 3000
sequence_length = 300
embedding_dimension = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
'''
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
'''
x_train = pad_sequences(x_train, maxlen=sequence_length)
x_test = pad_sequences(x_test, maxlen=sequence_length)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#构建基本句子分类器
def imdb_cnn():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension, input_length=sequence_length),
        layers.Conv1D(filters=50, kernel_size=5, strides=2, padding='valid'),
        layers.MaxPool1D(2, padding='valid'),
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )
    return model

model = imdb_cnn()
model.summary()

#训练模型
history = model.fit(x_train, y_train, batch_size=1, epochs=3, validation_split=0.1)

'''
#画图
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['traning', 'valiation'], loc='upper left')
plt.show()
'''

