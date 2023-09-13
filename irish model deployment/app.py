
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn

# Create flask app
flask_app = Flask(__name__, template_folder="/template")
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("template/index.html")

@flask_app.route("/predict", methods = ["GET","POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("template/index.html", prediction_text = ("The flower species is {}".format(prediction)))

if __name__ == "__main__":
    flask_app.run(debug=True)