from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.cars.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Collect form data from user
            input_data = {
                'nam': request.form['nam'],
                'Price': request.form['Price'],
                'Year': int(request.form['Year']),
                'Millage': request.form['Millage'],
                'Fuel': request.form['Fuel'],
                'Transmission': request.form['Transmission'],
                'Province': request.form['Province'],
                'Color': request.form['Color'],
                'Assembly': request.form['Assembly'],
                'Body Type': request.form['Body_Type'],
                'Ad Reference': request.form.get('Ad_Reference', ''),
                'Features': request.form.get('Features', ''),
                'Owner nam': request.form.get('Owner_nam', '')
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Prediction
            obj = PredictionPipeline()
            prediction = obj.predict(input_df)

            return render_template('results.html', prediction=str(prediction[0]))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong. Please check your input.'

    else:
        return render_template('index.html')

if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)