from transformers import pipeline
from Questioner import Questioner

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    context = 'Button is orange and says click me. Button is also small.'
    questioner = Questioner()
    answers = questioner.examine_context(context)
    print(answers)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
