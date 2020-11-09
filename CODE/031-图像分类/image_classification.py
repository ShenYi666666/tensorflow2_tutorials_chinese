# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 17:21
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
#1.图像类别
class_name = ['T-shirt/top', 'Trousure', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#2.探索数据
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
#3.处理数据
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''
train_images = train_images / 255.0
test_images = test_images / 255.0



