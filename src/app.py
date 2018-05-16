from flask import Flask, render_template, jsonify
from regression.workloadPrediction_LinearRegres import get_metrics
from classification.gaussianNBPredict import run_app

app = Flask(__name__)
app.debug = True

@app.route("/")
def index():
    #return render_template("index.html")
    print get_metrics()
    return jsonify(get_metrics())

@app.route("/gaussian")
def gaussian():
    return run_app()

if __name__ == '__main__':
    app.run(port=8080)