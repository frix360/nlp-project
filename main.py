from transformers import pipeline
from Questioner import Questioner
from ColorsModel import ColorsModel
from DataTransformer import DataTransformer
from HtmlGenerator import HtmlGenerator
import pandas


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    context = 'Button is orange and says click me. Button is also very big'
    questioner = Questioner()
    answers = questioner.examine_context(context)

    data = pandas.read_csv('set_0.csv')
    data.head()
    colors = data["name"]
    maxlen = 25
    num_classes = 28
    
    # model = ColorsModel(num_classes=num_classes, maxlen=maxlen, saved_model='./models/model_3.h5',
    #                     tokenizer_file_name="tokenizer.pickle")
    # model.fit(data=data, colors=colors)

    # color = 'tensorflow orange'
    # rgb = model.predict(color=color)
    # print(rgb)

    generator = HtmlGenerator()

    html = generator.generate_html(context)

    with open("output.html", "w+") as text_file:
        text_file.write(html)


    



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
