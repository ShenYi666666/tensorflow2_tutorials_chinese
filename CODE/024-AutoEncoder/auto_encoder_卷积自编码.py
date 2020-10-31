# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 15:06
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = tf.expand_dims(x_train.astype('float32'), -1) / 255.0
x_test = tf.expand_dims(x_test.astype('float32'), -1) / 255.0

print(x_train.shape, '    ', y_train.shape)
print(x_test.shape, '    ', y_test.shape)

inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='inputs')

code = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
code = layers.MaxPool2D((2, 2), padding='same')(code)
print(code.shape)

decoded = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(code)
decoded = layers.UpSampling2D((2, 2))(decoded)
print(decoded.shape)

outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
print(outputs.shape)

auto_encoder =keras.Model(inputs, outputs)

auto_encoder.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())

#keras.utils.plot_model(auto_encoder, show_shapes=True)

#提前停止
early_stop = keras.callbacks.EarlyStopping(patience=2, monitor='loss')
auto_encoder.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1, validation_freq=10, callbacks =[early_stop])

