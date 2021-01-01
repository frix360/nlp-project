from transformers import pipeline
from Questioner import Questioner
from ColorsModel import ColorsModel
from SizeModelLSTM import SizeModel
from DataTransformer import DataTransformer


class HtmlGenerator:
    def __init__(self):
        self.template = self.__init_template()
        self.questioner = Questioner()
        self.colors_model = ColorsModel(num_classes=28, maxlen=25, tokenizer_file_name="tokenizer.pickle",
                                        saved_model='./models/model_3.h5')
        self.size_model = SizeModel('lstm_tokenizer.pickle', saved_model='models/lstm_model.h5')
        self.answers = {
                           'color': None,
                           'size': None,
                           'text': None
                       },
        self.predicted_color = None,
        self.predicted_text = None
        self.predicted_size = None
        self.predicted_styles = None
        self.base_size = [256, 128]
        self.font_size = 1

    def __init_template(self):
        return '''
        <!DOCTYPE html>
        <html>
            <head> 
                <style> 
                    .button {
                          {--buttonStyles--}
                    }            
                </style>
            </head>
            <body>
                <button class='button'>{--buttonText--}</button>
            </body>
        </html>
        '''

    def generate_html(self, context):
        self.answers = self.questioner.examine_context(context)

        self.predicted_color = self.__get_predicted_color()
        self.predicted_size = self.__get_predicted_size()
        self.predicted_text = self.__get_predicted_text()

        replacer = {
            'buttonStyles': self.__generate_button_styles(self.predicted_color, self.predicted_size),
            'buttonText': self.predicted_text
        }

        return self.__replace_template(replacer)

    def __replace_template(self, replacer):
        result = self.template
        for key in replacer:
            str_to_replace = '{--%s--}' % key
            result = result.replace(str_to_replace, replacer[key])

        return result

    def __decide_font_color(self, color):
        if ((color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 186):
            return '#000000'
        return '#ffffff'

    def __generate_button_styles(self, color, size_ratio):
        formatted_color = f'rgb({color[0]}, {color[1]}, {color[2]})'
        font_color = self.__decide_font_color(color)

        return '''
                        background-color: {color};
                        min-width: {width}px;
                        height: {height}px;
                        color: {font_color};
                        font-size: {font_size}em'''.format(
                            color=formatted_color,
                            width=self.base_size[0] * (size_ratio), 
                            height=self.base_size[1] * (size_ratio),
                            font_size=self.font_size * (1/size_ratio), 
                            font_color=font_color
                        )
                

    def __get_predicted_color(self):
        return self.colors_model.predict(self.answers['color'].answer)

    def __get_predicted_size(self):
        if self.answers is None:
            return None

        return self.size_model.predict(self.answers['size'].answer)

    def __get_predicted_text(self):
        return self.answers['text'].answer
