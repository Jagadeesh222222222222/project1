import joblib
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load the fitted TfidfVectorizer
feature_extraction = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_mail = request.form['mail']

    # Convert text to feature vectors
    input_data_features = feature_extraction.transform([input_mail])

    # Make the prediction
    prediction = model.predict(input_data_features)

    # Return the prediction
    if prediction[0] == 1:
        result = 'Ham mail'
    else:
        result = 'Spam mail'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
