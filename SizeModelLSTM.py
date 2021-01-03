import tensorflow as tf
import datetime
import scipy.stats as stats
import numpy as np
import pickle
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape
from DataTransformer import DataTransformer

class SizeModel:
    def __init__(self, tokenizer_file_name, num_classes=28, maxlen=15, saved_model=None):
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
        log_dir = "logs/fit/size_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callback.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    def __initialize_model(self):
        # self.model = Sequential()
        # self.model.add(LSTM(128, return_sequences=True, input_shape=(self.maxlen, self.num_classes)))
        # self.model.add(LSTM(64))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dense(1, activation='sigmoid'))
        # self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])
        # self.model.summary()
        
        self.model = Sequential()
        self.model.add(LSTM(12, return_sequences=True, input_shape=(self.maxlen, self.num_classes)))
        self.model.add(LSTM(4))
        self.model.add(Dense(4, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])
        # self.model.summary()

    def predict(self, size):
        one_hot = self.data_transformer.transform_prediction_data(name=size)
        pred = self.model.predict(one_hot)[0]
        return pred[0]

    def fit(self, data, sizes, epohs=40, batch_size=32, validation_split=0.1):
        normalized_values, one_hot_names = self.data_transformer.transform_size_input_data(data=data, names=sizes)
        self.save_tokenizer('lstm_tokenizer.pickle')
        self.history = self.model.fit(x=one_hot_names, y=normalized_values, epochs=epohs,
                                      batch_size=batch_size, validation_split=validation_split,
                                      verbose=1, callbacks=self.callback)

    def __scale(self, n):
        MAX_RATIO = 2.1597913294039186
        return n * MAX_RATIO

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
