import json

from HtmlGenerator import HtmlGenerator
from flask import Flask, render_template, request

app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True


@app.route('/', methods=['GET', 'POST'])
def index():
    context = request.form.get('context') or ''

    html = None
    answers = None

    if context:
        generator = HtmlGenerator()
        data = generator.generate_button_html(context)

        html = data['html']
        answers = data['answers']

    return render_template('index.html', context=context, html=html, answers=answers)

# if __name__ == '__main__':
#     context = 'Button is orange and says click me. Button is also very big'
#
#     generator = HtmlGenerator()
#
#     html = generator.generate_html(context)
#
#     with open("output2.html", "w+") as text_file:
#         text_file.write(html)
