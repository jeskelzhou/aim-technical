from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("weight-prediction.html")
    elif request.method == 'POST':
        Gender = request.form.get('Gender')
        Height = request.form.get('Height')
        model = joblib.load("model-development/weight-prediction-using-linear-regression.pkl")
        result = model.predict([[Gender,Height]])

        result_weight="{:.2f}".format(result[0])
        Gender_dict = {
            '0': 'Female',
            '1': 'Male',
        }
        return render_template('weight-prediction.html', result=f"Weight prediction from gender {Gender_dict[Gender]} with Height {Height} Cm = ", result_weight=f'{result_weight} Kg')
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)