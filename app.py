from flask import Flask, render_template, request
import numpy as np
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('gait_model.keras')
# Load the scaler
scaler = joblib.load('scaler.pkl')

# Columns used during training
numerical_vars = [
    'SEX', 'AGE', 'HEIGHT', 'BODY_WEIGHT', 'BODY_MASS', 'SHOE_SIZE', 'ORTHOPEDIC_INSOLE',
    'STRIDE_LENGTH', 'STEP_FREQUENCY', 'BALANCE_SCORE', 'MUSCLE_STRENGTH',
    'JOINT_RANGE_OF_MOTION', 'INJURY_HISTORY', 'NEURO_CONDITION'
]

# Function to preprocess input data
def preprocess_data(data):
    data = scaler.transform(data)
    data = np.expand_dims(data, axis=2)  # Reshape for Conv1D input shape
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    form_data = request.form.to_dict()
    input_data = np.array([[
        float(form_data['sex']),
        float(form_data['age']),
        float(form_data['height']),
        float(form_data['body_weight']),
        float(form_data['body_mass']),
        float(form_data['shoe_size']),
        float(form_data['orthopedic_insole']),
        float(form_data['stride_length']),
        float(form_data['step_frequency']),
        float(form_data['balance_score']),
        float(form_data['muscle_strength']),
        float(form_data['joint_range_of_motion']),
        float(form_data['injury_history']),
        float(form_data['neuro_condition'])
    ]])

    # Detailed explanation variables
    explanation = []

    # Check for conditions that directly indicate gait impairment
    if float(form_data['neuro_condition']) == 1:
        explanation.append("Neurological condition present.")
        gait_impairment = "Gait impairment likely due to neurological condition"
        return render_template('result.html', gait_impairment=gait_impairment, explanation=explanation)

    if float(form_data['injury_history']) == 1:
        explanation.append("History of injury present.")
        gait_impairment = "Gait impairment likely due to injury history"
        return render_template('result.html', gait_impairment=gait_impairment, explanation=explanation)

    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make prediction
    readmission_prediction = model.predict(input_data)
    readmission_prediction_class = np.argmax(readmission_prediction, axis=1)[0]

    # Determine gait impairment
    if readmission_prediction_class == 1:
        gait_impairment = "Gait impairment likely based on model prediction"
    else:
        gait_impairment = "Gait impairment unlikely based on model prediction"

    explanation.append(f"Model prediction class: {readmission_prediction_class}")

    return render_template('result.html', gait_impairment=gait_impairment, explanation=explanation)