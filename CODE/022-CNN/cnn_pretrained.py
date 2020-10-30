# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 16:00
# @Author  : ShenYi
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

img = image.load_img('dog.jpg')
print(image.img_to_array(img).shape)




