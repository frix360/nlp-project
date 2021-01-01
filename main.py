from transformers import pipeline
from Questioner import Questioner
from ColorsModel import ColorsModel
from DataTransformer import DataTransformer
from HtmlGenerator import HtmlGenerator
import pandas
from nltk.corpus import wordnet as wn
import nltk
import numpy as np
import gensim.downloader as api
from scipy import spatial
from SizeModelLSTM import SizeModel
import re

def calculate_distance(dataSetI, dataSetII):
    return spatial.distance.cosine(dataSetI, dataSetII)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # context = 'Button is orange and says click me. Button is medium sized'

    # questioner = Questioner()
    # answers = questioner.examine_context(context)

    # wv_from_bin = api.load("glove-wiki-gigaword-200")

    # word = wv_from_bin['medium']
    # example_3 = wv_from_bin['sized']

    # lower_bound = wv_from_bin['small']
    # upper_bound = wv_from_bin['big']
    
    # print(calculate_distance(word, lower_bound))
    # print(calculate_distance(word, upper_bound))
    # print(calculate_distance(lower_bound, upper_bound))
    # print(calculate_distance(lower_bound, lower_bound))
    # print(calculate_distance(example_3, lower_bound))
    


    # print(answers)
    # words = answers['size'].answer.split(' ')
    

    # nltk.download('wordnet')
    # words = wn.synsets('tiny')
    # print(syn.definition())
    # print(syn.examples())

    # round(max(i.wup_similarity(n[0]) for i in g), 1)

    # print(word)
    # lower_bound = wn.synsets('small')
    # upper_bound = wn.synsets('big')

    # # ['huge button']
    # # 'small', 'big'
    # print(word.path_similarity(lower_bound))
    # print(word.path_similarity(upper_bound))
    
    # print([w.wup_similarity(b) for b in lower_bound for w in words])
    # print([w.wup_similarity(b) for b in upper_bound for w in words])
    
    # print(word.lch_similarity(lower_bound))
    # print(word.lch_similarity(upper_bound))

    # context = 'Button is orange and says click me. Button is also very big'
  

    # data = pandas.read_csv('set_0.csv')
    # data.head()
    # colors = data["name"]
    # maxlen = 25
    # num_classes = 28

    # generator = HtmlGenerator()

    # html = generator.generate_html(context)

    # with open("output.html", "w+") as text_file:
    #     text_file.write(html)
    
    model = SizeModel('lstm_tokenizer.pickle')
    # saved_model='models/lstm_model.h5'
    
    # print(model.predict('less undersize'))
    # print(model.predict('big'))
    data = pandas.read_csv('size_data.csv')
    features = data['name']
    labels = data['size']
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
    
    model.fit(data, processed_features)
    model.save_model('models/lstm_model.h5')
    
    print('very big', model.predict('very big'))
    print('biggest', model.predict('biggest'))
    print('big', model.predict('big'))
    print('huge', model.predict('huge'))
    print('small', model.predict('small'))
    print('very small', model.predict('very small'))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
