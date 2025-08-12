from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import re
import string
import unicodedata
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("french"))


app = Flask(__name__)

# chargement du pipeline de Random Forest
pipeline_rf = joblib.load("models/rf_credit_model.pkl")

# Pour le modèle NLP, on charge le pipeline
nlp_model = joblib.load("models/sentiment_model.pkl")


def nettoyer_texte(text):
    text = text.lower()  # minuscule
    text = re.sub(r"\d+", "", text)  # supprimer chiffres
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # supprimer ponctuation
    text = " ".join(
        [word for word in text.split() if word not in stop_words]
    )  # stopwords
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = (
        unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    )  # normalisation unicode
    text = re.sub(r"[^a-z\s]", "", text)  # Supprimer espaces multiples
    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_rf", methods=["POST"])
def predict_rf():
    # Récupération des données du formulaire index.html
    input_dict = {
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": request.form["ApplicantIncome"],
        "CoapplicantIncome": request.form["CoapplicantIncome"],
        "LoanAmount": request.form["LoanAmount"],
        "Loan_Amount_Term": request.form["Loan_Amount_Term"],
        "Credit_History": request.form["Credit_History"],
        "Property_Area": request.form["Property_Area"],
    }
    # Création d'un DataFrame à partir du dictionnaire

    features = pd.DataFrame([input_dict])

    # prediction avec le pipeline
    pred = pipeline_rf.predict(features)[0]
    proba = pipeline_rf.predict_proba(features)[0][1]

    # Affichage du message
    if pred == 1:
        result_message = " Eligible"
    else:
        result_message = "Pas Eligible"
    # Affichage du résultat dans index.html
    return render_template("index.html", rf_prediction=result_message)


@app.route("/predict_nlp", methods=["POST"])
def predict_nlp():
    result_nlp = None

    if request.method == "POST":
        raw_text = request.form["text_nlp"]

        text = nettoyer_texte(raw_text)

        if not text:
            render_template(
                "index.html",
                prediction_nlp="Veuillez entrer un texte pour l'analyse NLP.",
            )

        else:
            # Utilisation du modèle NLP pour prédire le sentiment
            pred = nlp_model.predict([text])[0]
            result_nlp = f" sentiment : {pred}"
        # if pred == 1:
        # result_nlp = "Le sentiment est positif."
        # else:
        # result_nlp = "Le sentiment est négatif."

        return render_template("index.html", prediction_nlp=result_nlp)


@app.route("/predict_llm", methods=["POST"])
def predict_llm():
    if request.method == "POST":
        text = request.form["text_llm"]
        result = fake_llm_predict(text)
        return render_template("index.html", prediction_llm=result)


if __name__ == "__main__":
    app.run(debug=True)
