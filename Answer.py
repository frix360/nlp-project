import json


class Answer:
    def __init__(self, answer):
        self.score = answer['score']
        self.start = answer['start']
        self.end = answer['end']
        self.answer = answer['answer']
        self.question = answer['question']

    def __to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __str__(self):
        return self.__to_json()

    def __repr__(self):
        return self.__to_json()
