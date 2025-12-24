from flask import Flask, render_template, request
import pickle
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# Load models
tfidf_class = pickle.load(open("models/tfidf_class.pkl", "rb"))
svm = pickle.load(open("models/svm_classifier.pkl", "rb"))

tfidf_reg = pickle.load(open("models/tfidf_reg.pkl", "rb"))
ridge = pickle.load(open("models/ridge_regressor.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    score = None
    submitted_text = None
    word_count = None
    char_count = None

    if request.method == "POST":
        submitted_text = request.form["text"]
        word_count = len(submitted_text.split())
        char_count = len(submitted_text)

        # Classification
        X_class = tfidf_class.transform([submitted_text])
        prediction = svm.predict(X_class)[0]

        # Regression
        text_len = len(submitted_text)
        X_reg_text = tfidf_reg.transform([submitted_text])
        X_reg = hstack([X_reg_text, [[text_len]]])
        score = round(ridge.predict(X_reg)[0], 2)

    return render_template("index.html",
                           prediction=prediction,
                           score=score,
                           submitted_text=submitted_text,
                           word_count=word_count,
                           char_count=char_count
                           )
if __name__ == "__main__":
    app.run(debug=True)