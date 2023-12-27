from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('Heart_Disease_Prediction.pkl')

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Convert form data to numeric types
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    oldpeak = float(request.form["oldpeak"])
    thalach = float(request.form["thalach"])
    fbs = float(request.form["fbs"])
    exang = float(request.form["exang"])
    slope = float(request.form["slope"])
    cp = float(request.form["cp"])
    thal = float(request.form["thal"])
    ca = float(request.form["ca"])
    restecg = float(request.form["restecg"])

    arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    pred = model.predict(arr)

    if pred == 0:
        res_val = "NO HEART PROBLEM"
    else:
        res_val = "HEART PROBLEM"

    return render_template('home.html', prediction_text='PATIENT HAS {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
