from flask import Flask, render_template, jsonify
from regression.workloadPrediction_LinearRegres import loadPredictionDataSet, statsSummary, getMetrics

app = Flask(__name__)
app.debug = True

@app.route("/")
def index():
    #return render_template("index.html")
    return jsonify(getMetrics())

if __name__ == '__main__':
    app.run(port=8080)