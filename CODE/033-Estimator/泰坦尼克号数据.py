# -*- coding: utf-8 -*-
# @Time    : 2020/11/9 14:15
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd

#1.加载泰坦尼克号数据
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

tf.random.set_seed(123)

#2.探索数据
#print(dftrain.head())
#print(dftrain.describe())


@todo




