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
            'smallest': [10, 20, -1],
            'smaller': [20, 30, -1],
            'small': [30, 40, -1],
            'average': [40, 60, 0],
            'big': [60, 80, 1],
            'bigger': [80, 100, 1],
            'biggest': [100, 120, 1]
        }

        adjectives = {
            'very': 30,
            'less': -10,
            'more': 10,
            'much': 40,
            'really': 60
        }


        data = {}


        for word in words.keys():
            all_words = self.get_synonims(word) + [ word ]
            min_bound = words[word][0] 
            max_bound = words[word][1] 
            positivity = words[word][2]
            for w in all_words:              
                base_ratio = random.uniform(min_bound, max_bound)
                data[w] = int(base_ratio)

                for a in adjectives.keys():
                    new_word = a + ' ' + w
                    ratio = 0
                    if positivity >= 0:
                        ratio = base_ratio + adjectives[a]
                    else:
                        ratio = base_ratio - adjectives[a]
                    
                    data[new_word] = int(ratio)

        return data


    def generate_file(self, data, filename='size_data.csv'):
        with open(filename, 'w') as file:
            file.write('name,size\n')
            for word in data.keys():
                line = word + ',' + str(data[word]) + '\n'
                file.write(line)


gen = TrainingSetGenerator()
data = gen.generate_training_set()
gen.generate_file(data)