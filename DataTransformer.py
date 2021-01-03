from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import preprocessing
from keras.utils import np_utils
import numpy as np
import re


class DataTransformer:
    def __init__(self, maxlen, num_classes):
        self.tokenizer = Tokenizer(char_level=True)
        self.maxlen = maxlen
        self.num_classes = num_classes

    def __norm(self, value):
        return value / 255.0

    def __norm_size(self, value):
        return (value - value.min()) / (value.max() - value.min())

    def __process_data(self, names):
        processed_features = []
        for sentence in range(0, len(names)):
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', str(names[sentence]))
            # remove all single characters
            processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
            # Removing prefixed 'b'
            processed_feature = re.sub(r'^b\s+', '', processed_feature)
            # Converting to Lowercase
            processed_feature = processed_feature.lower()
            processed_features.append(processed_feature)
        return processed_features

    def transform_size_input_data(self, data, names):
        names = self.__process_data(names)
        self.tokenizer.fit_on_texts(names) # [1: 'I', 2, 3, 4]
        tokenized = self.tokenizer.texts_to_sequences(names)  # [sen, sen] -> [[1, 2, 3, 4], [4, 2, 3, 5]]
        padded_names = preprocessing.sequence.pad_sequences(tokenized, maxlen=self.maxlen)

        one_hot_names = np_utils.to_categorical(padded_names)
        # self.num_classes = one_hot_names.shape[-1]
        normalized_values = np.column_stack([self.__norm_size(data['size'])])
        return normalized_values, one_hot_names

    def transform_colors_input_data(self, data, colors):
        self.tokenizer.fit_on_texts(colors) # [1: 'I', 2, 3, 4]
        tokenized = self.tokenizer.texts_to_sequences(colors)  # [sen, sen] -> [[1, 2, 3, 4], [4, 2, 3, 5]]
        padded_colors = preprocessing.sequence.pad_sequences(tokenized, maxlen=self.maxlen)

        one_hot_names = np_utils.to_categorical(padded_colors)
        self.num_classes = one_hot_names.shape[-1]
        normalized_values = np.column_stack([self.__norm(data["red"]),
                                             self.__norm(data["green"]),
                                             self.__norm(data["blue"])])
        return normalized_values, one_hot_names

    def transform_prediction_data(self, name):
        name = name.lower()
        tokenized = self.tokenizer.texts_to_sequences([name])
        padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=self.maxlen)
        return np.array(np_utils.to_categorical(padded, self.num_classes))
