# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 10:16
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

'''
Keras函数式API
该方法比Sequential方法灵活：可以处理非线性模型，具有共享层的模型，及多输入、多输出模型

'''
#1.构建简单的网络
#1.1 创建网络

inputs = tf.keras.Input(shape=(784, ), name='img')

h1 = layers.Dense(32, activation='relu')(inputs)
h2 = layers.Dense(32, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h2)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist')

model.summary()
#keras.utils.plot_model(model, 'mnist_model.png')
#keras.utils.plot_model(model, 'model_info.png', show_shapes=True)

#1.2 训练、验证及测试
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float')/255
model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
print('test loss:', score[0])
print('test acc:', score[1])

#1.3 模型保存和序列化
model.save('model_save.h5')
del model
model = keras.models.load_model('model_save.h5')

#2.使用共享网络创建多个模型
#自编码器网络结构
#编码器
encode_input = keras.Input(shape=(28, 28, 1), name='img')
h1 = layers.Conv2D(16, 3, activation='relu')(encode_input)
h1 = layers.Conv2D(32, 3, activation='relu')(h1)
h1 = layers.MaxPool2D(3)(h1)
h1 = layers.Conv2D(32, 3, activation='relu')(h1)
h1 = layers.Conv2D(16, 3, activation='relu')(h1)
encode_output = layers.GlobalMaxPool2D()(h1)

encode_model = keras.Model(inputs=encode_input, outputs=encode_output, name='encoder')
encode_model.summary()

#解码器
h2 = layers.Reshape((4, 4, 1))(encode_output)
h2 = layers.Conv2DTranspose(16, 3, activation='relu')(h2)
h2 = layers.Conv2DTranspose(32, 3, activation='relu')(h2)
h2 = layers.UpSampling2D(3)(h2)
h2 = layers.Conv2DTranspose(16, 3, activation='relu')(h2)

decode_output = layers.Conv2DTranspose(1, 3, activation='relu')(h2)

autoencoder = keras.Model(inputs=encode_input, outputs = decode_output, name='autoencoder')
autoencoder.summary()

#模型嵌套集成
def get_model():
    inputs = keras.Input(shape=(128, ))
    outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
    return keras.Model(inputs, outputs)

model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128, ))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)

outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs, outputs)

#3.复杂网络结构构建
#3.1 多输入与多输出网络

#构建一个根据定制标题、内容和标签， 预测票证优先级和执行部门的网络
#超参数
num_words = 2000
num_tags = 12
num_department = 4

#输入
body_input = keras.Input(shape=(None,), name='body')
title_input = keras.Input(shape=(None,), name='title')
tag_input = keras.Input(shape=(num_tags, ), name='tag')

#嵌入层
body_feat = layers.Embedding(num_words, 64)(body_input)
title_feat = layers.Embedding(num_words, 64)(title_input)

#特征提取层
body_feat = layers.LSTM(32)(body_feat)
title_feat = layers.LSTM(128)(title_feat)
features = layers.concatenate([title_feat, body_feat, tag_input])

#分类层
priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(features)
department_pred = layers.Dense(num_department, activation='softmax', name='department')(features)

#构建模型
model = keras.Model(inputs = [body_input, title_input, tag_input], outputs= [priority_pred, department_pred])
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss={'priority':keras.losses.binary_crossentropy, 'department':keras.losses.categorical_crossentropy}, loss_weights=[1., 0.2])

#4.共享网络层

#5.模型复用
from tensorflow.keras.applications import VGG16
import numpy as np

vgg16 = VGG16()
#获取中间结构输出
feature_list = [layers.output for layer in vgg16.layers]
#将其作为新模型输出
feat_ext_model = keras.Model(inputs=vgg16.input, outputs=feature_list)

img = np.random.random((1, 224, 224, 3)).astype('float32')

#用于提取特征
e





