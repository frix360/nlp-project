from transformers import pipeline
from Questions import Questions
from Answer import Answer


class Questioner:
    def __init__(self):
        self.answerer = pipeline('question-answering')
        self.questions_manager = Questions()
        self.context = None

    def examine_context(self, context):
        self.context = context
        types = self.questions_manager.get_types()
        answers = []
        for question_type in types:
            questions = self.questions_manager.get_questions_by_type(question_type)
            type_answers = self.__ask_questions(questions)
            best_answer = Questioner.get_best_answer(type_answers)
            answers.append(best_answer)
        return answers

    def __ask_questions(self, questions):
        self.__check_for_none_context()
        answers = [self.__ask_question(question) for question in questions]
        return answers

    def __ask_question(self, question):
        raw_answer = self.answerer({
            'question': question,
            'context': self.context
        })
        raw_answer['question'] = question
        answer = Answer(raw_answer)

        return answer

    def __check_for_none_context(self):
        if self.context is None:
            raise Exception("Context cannot be None")

    @staticmethod
    def get_best_answer(answers):
        best = answers[0]
        best_score = answers[0].score
        for answer in answers:
            if answer.score > best_score:
                best = answer

        return best
