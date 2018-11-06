from driver import main
from flask import Flask, Response, render_template, request, redirect, send_file
from werkzeug import secure_filename
import time as time

import os.path

app = Flask(__name__)

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)

@app.route('/')
def index():
    content = get_file('web/index.html')
    return Response(content, mimetype="text/html")

@app.route('/render/<id>')
def render(id):
    print(id)
    content = get_file('web/render.html')
    return Response(content, mimetype="text/html")



@app.route('/uploader', methods=['GET', 'POST'])
def uploadr_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('temp/' + str(secure_filename(f.filename)))
        return redirect('/render/' + str(secure_filename(f.filename)))



@app.route('/page')
def get_page():
    return send_file('web/progress.html')

@app.route('/progress')
def progress():
    def generate():
        # x = 0
        # while x < 100:
        #     print x
        #     x = x + 10
        #     time.sleep(0.2)
        #     yield "data:" + str(x) + "\n\n"
        main()
    return Response(generate(), mimetype= 'text/event-stream')


if __name__ == '__main__':
    app.run()