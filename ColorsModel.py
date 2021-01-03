import tensorflow as tf
import datetime
import scipy.stats as stats
import numpy as np
import pickle
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape
from DataTransformer import DataTransformer


class ColorsModel:
    def __init__(self, tokenizer_file_name, num_classes=28, maxlen=25, saved_model=None):
        self.maxlen = maxlen
        self.num_classes = num_classes
        self.history = None
        self.__initialize_model()
        self.callback = []
        self.__initialize_callback()
        self.data_transformer = DataTransformer(maxlen, num_classes)
        if saved_model is not None:
            self.data_transformer.tokenizer = self.load_tokenizer(tokenizer_file_name)
            self.__load_saved_model(saved_model)

    def __load_saved_model(self, saved_model):
        self.model.load_weights(saved_model)

    def __initialize_callback(self):
        log_dir = "logs/fit/color_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callback.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    def __initialize_model(self):
        self.model = Sequential()
        self.model.add(LSTM(256, return_sequences=True, input_shape=(self.maxlen, self.num_classes)))
        self.model.add(LSTM(128))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])
        # self.model.summary()

    def __scale(self, n):
        return int(n * 255)

    def predict(self, color):
        one_hot = self.data_transformer.transform_prediction_data(name=color)
        pred = self.model.predict(one_hot)[0]
        return [self.__scale(pred[0]), self.__scale(pred[1]), self.__scale(pred[2])]

    def fit(self, data, colors, epohs=40, batch_size=32, validation_split=0.1):
        normalized_values, one_hot_names = self.data_transformer.transform_colors_input_data(data=data, colors=colors)
        self.save_tokenizer('tokenizer_2.pickle')
        self.history = self.model.fit(x=one_hot_names, y=normalized_values, epochs=epohs, batch_size=batch_size,
                                      validation_split=validation_split, verbose=1, callbacks=self.callback)

    def save_model(self, file_name):
        self.model.save_weights(file_name)
    
    def save_tokenizer(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self.data_transformer.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_tokenizer(self, file_name):
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)
    
    def save_progress(self, model_file_name, tokenizer_file_name):
        self.save_model(model_file_name)
        self.save_tokenizer(tokenizer_file_name)
