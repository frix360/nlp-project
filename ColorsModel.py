import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape
from keras.utils import np_utils

import scipy.stats as stats

import numpy as np
import pandas


def norm(value):
    return value / 255.0


def scale(n):
    return int(n * 255)


def predict(name):
    name = name.lower()
    tokenized = t.texts_to_sequences([name])
    padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=maxlen)
    one_hot = np_utils.to_categorical(padded, num_classes=num_classes)
    pred = model.predict(np.array(one_hot))[0]
    r, g, b = scale(pred[0]), scale(pred[1]), scale(pred[2])
    print(name + ',', 'R,G,B:', r, g, b)


data = pandas.read_csv('set_0.csv')

data.head()

names = data["name"]

maxlen = 25
t = Tokenizer(char_level=True)
t.fit_on_texts(names)  # [1: 'I', 2, 3, 4]
tokenized = t.texts_to_sequences(names)  # [sen, sen] -> [[1, 2, 3, 4], [4, 2, 3, 5]]
padded_names = preprocessing.sequence.pad_sequences(tokenized, maxlen=maxlen)

one_hot_names = np_utils.to_categorical(padded_names)
num_classes = one_hot_names.shape[-1]
normalized_values = np.column_stack([norm(data["red"]), norm(data["green"]), norm(data["blue"])])

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, num_classes)))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['acc'])
model.summary()

# model.load_weights('./models/model_2.h5')

# color_name = 'Keras red'
# predict(color_name)

history = model.fit(one_hot_names, normalized_values,
                    epochs=40,
                    batch_size=32,
                    validation_split=0.1)
#

model.save_weights('./models/model_3.h5')

color_name = 'Keras red'
predict(color_name)
