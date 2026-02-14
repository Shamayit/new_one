from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("disease.pkl","rb") as f:
    bundle = pickle.load(f)

model_disease = bundle["model_disease"]
model_medicine = bundle["model_medicine"]
label_disease = bundle["label_disease"]
label_medicine = bundle["label_medicine"]
vectorizer = bundle["vectorizer"]

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form["symptoms"]
    X = vectorizer.transform([symptoms])

    d_enc = model_disease.predict(X)[0]
    disease = label_disease.inverse_transform([d_enc])[0]

    m_enc = model_medicine.predict(X)[0]
    medicine = label_medicine.inverse_transform([m_enc])[0]

    meds = [medicine]

    return render_template("prescription.html",
                           disease=disease,
                           symptoms=symptoms,
                           meds=meds)

if __name__ == "__main__":
    app.run(debug=True)
