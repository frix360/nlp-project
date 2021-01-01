import numpy as np 
import pandas as pd 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


class SizeModel:
    def __init__(self, tokenizer_file_name, max_features=2500, min_df=7, max_df=0.8, n_estimators=200, random_state=0, saved_model=None):
        self.vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
        self.model = RandomForestClassifier(n_estimators=200, random_state=0)

        if saved_model is not None:
            self.vectorizer = self.load_tokenizer(tokenizer_file_name)
            self.__load_saved_model(saved_model)

    def fit(self, data, labels):
        processed_features = self.vectorizer.fit_transform(data).toarray()
        self.model.fit(processed_features, labels)

    def __scale(self, n):
        MAX_RATIO = 10
        return int(n * MAX_RATIO)

    def predict(self, data):
        processed_features = self.vectorizer.fit_transform(data).toarray()
        return self.__scale(self.model.predict(processed_features))
    
    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)
    
    def save_progress(self, model_file_name, tokenizer_file_name):
        self.save_model(model_file_name)
        self.save_tokenizer(tokenizer_file_name)
    
    def __load_saved_model(self, saved_model):
        with open(saved_model, 'rb') as f:
            self.model = pickle.load(f)

    def save_tokenizer(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_tokenizer(self, file_name):
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)



data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
print(airline_tweets.head())

features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

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


X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

model = SizeModel('scipy_vectorizer.pickle')
model.fit(X_train, y_train)
model.save_model('scipy_model_forest.pk')
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

# data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
# airline_tweets = pd.read_csv(data_source_url)
# print(airline_tweets.head())

# features = airline_tweets.iloc[:, 10].values
# labels = airline_tweets.iloc[:, 1].values

# processed_features = []

# for sentence in range(0, len(features)):
#     # Remove all the special characters
#     processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

#     # remove all single characters
#     processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

#     # Remove single characters from the start
#     processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

#     # Substituting multiple spaces with single space
#     processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

#     # Removing prefixed 'b'
#     processed_feature = re.sub(r'^b\s+', '', processed_feature)

#     # Converting to Lowercase
#     processed_feature = processed_feature.lower()

#     processed_features.append(processed_feature)

# # TODO: decide min_df value
# vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
# processed_features = vectorizer.fit_transform(processed_features).toarray()

# X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

# text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
# text_classifier.fit(X_train, y_train)

# predictions = text_classifier.predict(X_test)

# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))
# print(accuracy_score(y_test, predictions))