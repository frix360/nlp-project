from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import preprocessing
from keras.utils import np_utils
import numpy as np


class DataTransformer:
    def __init__(self, maxlen, num_classes):
        self.tokenizer = Tokenizer(char_level=True)
        self.maxlen = maxlen
        self.num_classes = num_classes

    def __norm(self, value):
        return value / 255.0

    def transform_input_data(self, data, colors):
        self.tokenizer.fit_on_texts(colors) # [1: 'I', 2, 3, 4]
        tokenized = self.tokenizer.texts_to_sequences(colors)  # [sen, sen] -> [[1, 2, 3, 4], [4, 2, 3, 5]]
        padded_colors = preprocessing.sequence.pad_sequences(tokenized, maxlen=self.maxlen)

        one_hot_names = np_utils.to_categorical(padded_colors)
        normalized_values = np.column_stack([self.__norm(data["red"]),
                                             self.__norm(data["green"]),
                                             self.__norm(data["blue"])])
        return normalized_values, one_hot_names

    def transform_prediction_data(self, name):
        name = name.lower()
        tokenized = self.tokenizer.texts_to_sequences([name])
        padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=self.maxlen)
        return np.array(np_utils.to_categorical(padded, self.num_classes))
