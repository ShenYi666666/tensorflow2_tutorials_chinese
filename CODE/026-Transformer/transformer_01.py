# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 16:07
# @Author  : ShenYi
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.layers as layers
import time
import numpy as np
import matplotlib.pyplot as plt

example, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
#1.数据输入pipeline
#2.位置嵌入
#3.掩码
#4.Scaled dot product attension
#5.Multi-Head Attension
#6.解码器和编码器
#7.创建transformer
#8.实验设置
#9.训练和保持模型
#10.



