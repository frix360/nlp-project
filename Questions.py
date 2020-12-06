class Questions:
    def __init__(self):
        self.questions = {}
        self.__init_questions()

    def __init_questions(self):
        self.questions = {
            'color': [
                'What is the color of the button',
                'What is the button\'s color',
                'Color of the button',
                'Button\'s color is'
            ],
            'size': [
                'What is the size of a button',
                'Button size is',
                'What is the button\'s size',
            ],
            'text': [
                'What does the button say',
                'What is the text of the button',
                'What is written on the button'
            ]
        }

    def get_types(self):
        return list(self.questions.keys())

    def get_questions_by_type(self, question_type):
        return self.questions[question_type]
