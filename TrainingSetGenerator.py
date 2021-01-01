from nltk.corpus import wordnet as wn
import nltk
import itertools
import random
from PyDictionary import PyDictionary

class TrainingSetGenerator:

    def get_synonims(self, word):
        dictionary=PyDictionary()
        return dictionary.synonym(word)

    def generate_training_set(self):
        random.seed(0)
        words = {
            'smallest': [0, 0.2, -1],
            'smaller': [0.2, 0.4, -1],
            'small': [0.2, 0.4, -1],
            'average': [0.4, 0.6, 0],
            'big': [0.6, 0.8, 1],
            'bigger': [0.8, 1.0, 1],
            'biggest': [1.0, 1.2, 1]
        }

        adjectives = {
            'very': 1.8,
            'less': 0.7,
            'more': 1.2,
            'much': 1.5,
            'really': 1.6
        }


        data = {}


        for word in words.keys():
            all_words = self.get_synonims(word) + [ word ]
            min_bound = words[word][0] 
            max_bound = words[word][1] 
            positivity = words[word][2]
            for w in all_words:              
                base_ratio = random.uniform(min_bound, max_bound)
                data[w] = round(base_ratio, 2)

                for a in adjectives.keys():
                    new_word = a + ' ' + w
                    ratio = 0
                    if positivity >= 0:
                        ratio = base_ratio * adjectives[a]
                    else:
                        ratio = base_ratio / adjectives[a]
                    
                    data[new_word] = round(ratio, 2)

        return data


    def generate_file(self, data, filename='size_data.csv'):
        with open(filename, 'w') as file:
            file.write('name,size \n')
            for word in data.keys():
                line = word + ',' + str(data[word]) + '\n'
                file.write(line)


gen = TrainingSetGenerator()
data = gen.generate_training_set()
gen.generate_file(data)