# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 14:26
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(y_train[0:5])

