from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('/index.html')



    

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        # Get form data
        to_predict_list = request.form.to_dict()

        # check if 'prev_contacted' is checked
        prev_contacted = 'prev_contacted' in to_predict_list

        # Call preprocessDataAndPredict with the appropriate model based on 'prev_contacted'
        try:
            if prev_contacted:
                prediction = preprocessDataAndPredict(to_predict_list, model_path="prediction_pipeline_prev_y.pkl")
            else:
                prediction = preprocessDataAndPredict(to_predict_list, model_path="prediction_pipeline_prev_n.pkl")
            
            # Pass prediction to the template
            return render_template('predict.html', prediction=prediction)
        except ValueError:
            return "Please Enter valid values"
        pass
    pass
  
def preprocessDataAndPredict(feature_dict, model_path):
    # Convert feature_dict to a DataFrame
    test_data = {k: [v] for k, v in feature_dict.items()}
    test_data = pd.DataFrame(test_data)

    # Load the appropriate trained model
    with open(model_path, "rb") as file:
        trained_model = joblib.load(file)
    
    # Make prediction
    predict = trained_model.predict(test_data)

    return predict
    
    

if __name__ == '__main__':
    app.run(debug=True)