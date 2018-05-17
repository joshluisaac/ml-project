from flask import Flask, render_template, jsonify
from regression.linear_regression import get_metrics
from classification.gaussianNBPredict import run_app
from time import sleep

app = Flask(__name__)
app.debug = True

@app.route("/regression")
def index():
    #return render_template("index.html")
    #print get_metrics()
    return jsonify(get_metrics())

@app.route("/gaussian")
def gaussian():
    #sleep(500)
    return jsonify(run_app())

if __name__ == '__main__':
    app.run(port=9099)