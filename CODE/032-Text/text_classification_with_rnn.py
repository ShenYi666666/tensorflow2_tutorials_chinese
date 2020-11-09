# -*- coding: utf-8 -*-
# @Time    : 2020/11/9 11:31
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow,keras.layers as layers
import tensorflow_datasets as tfds

datasets, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

#1.获取训练集， 测试集
train_data, test_data = datasets['train'], datasets['test']

tokenizer = info.features['text'].encoder
print(tokenizer.vocab_size)


