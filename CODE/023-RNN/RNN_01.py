# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 17:29
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

num_word = 30000
max_len = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
print(x_train.shape, '    ', y_train.shape)
print(x_test.shape, '     ', y_test.shape)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, max_len, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, max_len, padding='post')

print(x_train.shape, '      ', y_train.shape)
print(x_test.shape, '      ', y_test.shape)

def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=max_len),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )
    return model
model = lstm_model()
model.summary()

