from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.diabeties.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            gender =int(request.form['gender'])
            age =float(request.form['age'])
            hypertension =float(request.form['hypertension'])
            heart_disease =float(request.form['heart_disease'])
            smoking_history =int(request.form['smoking_history'])
            bmi =float(request.form['bmi'])
            HbA1c_level =float(request.form['HbA1c_level'])
            blood_glucose_level =float(request.form['blood_glucose_level'])
         
            data = [gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]
            data = np.array(data).reshape(1, 8)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)