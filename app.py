# -- coding: utf-8 --
"""
Created on Sat Nov  2 17:15:35 2024

@author: CH Vishnu Vardhan
22835A6602
CSE(AIML)
IV rth Year First Semester
GURUNANAK INSTITUTE OF TECHNOLOGY,Ibrahimpatanam
"""

# 1. Importing Libraries
import random
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Loading Data
data = pd.read_csv('diabetes.csv')  # Make sure the CSV file is in the same directory as this script

# 3. Splitting Data into Train and Test Sets
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training the Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
model = LogisticRegression()
model.fit(X_train_scaled, y_train)  # Train the model

# 5. Making Predictions
X_test_scaled = scaler.transform(X_test)  # Only transform the test data
predictions = model.predict(X_test_scaled)  # Make predictions on the test set
probabilities = model.predict_proba(X_test_scaled)[:, 1] * 100  # Get probabilities of positive class

# 6. Visualize Results
plt.figure(figsize=(10, 6))
sns.countplot(x='Outcome', data=data)
plt.title('Distribution of Diabetes Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Diabetic', 'Diabetic'])
plt.show()

# 7. New Test Case (Flask Application)
app = Flask(_name_)

# List of health tips
health_tips = [
    "Stay active! Aim for at least 30 minutes of physical activity most days.",
    "Eat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
    "Monitor your blood sugar levels regularly.",
    "Stay hydrated by drinking plenty of water throughout the day.",
    "Get enough sleep! Aim for at least 7-8 hours of sleep each night."
]

@app.route('/')
def index():
    random_tip = random.choice(health_tips)  # Select a random health tip
    return '''
    <html>
    <head>
        <title>Diabetes Prediction </title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #e9f5ff;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            h2 {{ color: #333; }}
            form {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); width: 300px; }}
            input[type="number"] {{ width: calc(100% - 20px); padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; }}
            button {{ padding: 10px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            #result {{ margin-top: 20px; font-size: 18px; }}
            .tip {{ margin-top: 30px; font-style: italic; color: #555; }}
        </style>
    </head>
    <body>
        <h2>Diabetes Prediction Form</h2>
        <h6>by   CH Vishnu Vardhan<br>22835A6602<br>CSE(AIML)<br> IV rth Year First Semester<br>GURUNANAK INSTITUTE OF TECHNOLOGY,Ibrahimpatanam<br>
        <form id="diabetesForm" method="post" action="/predict">
            <label for="pregnancies">Pregnancies:</label>
            <input type="number" name="Pregnancies" required><br>
            <label for="glucose">Glucose:</label>
            <input type="number" name="Glucose" required><br>
            <label for="bloodPressure">Blood Pressure:</label>
            <input type="number" name="BloodPressure" required><br>
            <label for="skinThickness">Skin Thickness:</label>
            <input type="number" name="SkinThickness" required><br>
            <label for="insulin">Insulin:</label>
            <input type="number" name="Insulin" required><br>
            <label for="bmi">BMI:</label>
            <input type="number" step="0.1" name="BMI" required><br>
            <label for="diabetesPedigreeFunction">Diabetes Pedigree Function:</label>
            <input type="number" step="0.001" name="DiabetesPedigreeFunction" required><br>
            <label for="age">Age:</label>
            <input type="number" name="Age" required><br>
            <button type="submit">Predict Diabetes</button><br>
        </form>
        <div id="result"></div>
        <div class="tip">Health Tip: {random_tip}</div>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
        $(document).ready(function() {{
            $('#diabetesForm').on('submit', function(event) {{
                event.preventDefault(); // Prevent form submission
                const formData = $(this).serializeArray();
                const patientData = {};
                formData.forEach(item => {{
                    patientData[item.name] = item.value;
                }});
                $.ajax({{
                    url: '/predict',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify(patientData),
                    success: function(data) {{
                        $('#result').html(Prediction (1: Diabetic, 0: Not Diabetic): ${data.prediction}, Probability of having diabetes: ${data.probability.toFixed(2)}%);
                    }},
                    error: function() {{
                        $('#result').html('An error occurred while predicting.');
                    }}
                }});
            }});
        }});
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    # Scale the input data
    scaled_data = scaler.transform([input_data])

    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] * 100  # Probability of having diabetes

    # Return prediction result as JSON
    return jsonify({
        'prediction': int(prediction[0]),  # 1 for diabetic, 0 for not diabetic
        'probability': probability
    })

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5001)  # Change to 127.0.0.1 if testing locally
