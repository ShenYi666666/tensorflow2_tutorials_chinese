# -*- coding: utf-8 -*-
# @Time    : 2020/11/9 10:14
# @Author  : ShenYi
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
#1.载入数据


vocal_size = 1000
(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data()
print(train_x[0])
print(train_x[1])

word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<SRART>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = {v:k for k, v in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_x[0]))

max_len = 500
train_x = keras.preprocessing.sequence.pad_sequences(train_x, value=word_index['<PAD>'], padding='post', maxlen=max_len)
test_x = keras.preprocessing.sequence.pad_sequences(test_x, value=word_index['<PAD>'], padding='post', maxlen=max_len)

#2.构建模型
embedding_dim = 16
model = keras.Sequential([
    layers.Embedding(vocal_size, embedding_dim, input_length= max_len),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=30, batch_size=16, validation_split=0.1)

#3.结果可视化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc , 'bo', label = 'Traning_acc')
plt.plot(epochs, val_acc, 'b', label = 'Val_acc')
plt.title('Training and Val acc')
plt.xlabel('epochs')
plt.ylabel('Acc')
plt.legend(loc= 'lower_right')
plt.figure(figsize=(16, 9))
plt.show()