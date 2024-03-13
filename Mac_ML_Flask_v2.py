from flask import Flask, request, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model and preprocessor pipeline
model_pipeline = joblib.load('Mac v6 model.joblib')

@app.route('/predict-file', methods=['POST'])
def predict_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part in the request', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected for uploading', 400

    # Use Pandas to read the Excel file
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return f'Error reading the Excel file: {str(e)}', 500

    # Preprocess and predict using the loaded model
    try:
        predictions = model_pipeline.predict(df)
        df['Mac going out level'] = predictions
    except Exception as e:
        return f'Error during prediction: {str(e)}', 500

    # Save the modified DataFrame to a new Excel file
    output_filename = 'Mac predictions.xlsx'
    df.to_excel(output_filename, index=False)

    # Return the Excel file with predictions as a download
    return send_file(output_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)