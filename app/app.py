from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from models.text_preprocessor import clean_text
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('app/models/cyberbullying_model.pkl')
vectorizer = joblib.load('app/models/vectorizer.pkl')

# Load dataset to extract offensive words
data = pd.read_csv('data/raw/cyberbullying_data.csv', encoding='ISO-8859-1')
data['cleaned_text'] = data['text'].apply(clean_text)
offensive_words = set()

for text in data[data['label'] == 1]['cleaned_text']:
    offensive_words.update(text.split())

# Function to highlight offensive words
def highlight_offensive_words(input_text):
    """
    Identify offensive words from the input text that match known offensive words.
    """
    cleaned_text = clean_text(input_text)
    words = cleaned_text.split()
    detected_words = [word for word in words if word in offensive_words]
    return detected_words

# Prediction function
def predict_text(input_text):
    """
    Predict cyberbullying content and detect offensive words or phrases.
    """
    cleaned_text = clean_text(input_text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)
    
    offensive_detected = highlight_offensive_words(input_text)
    result = "Cyberbullying" if prediction[0] == 1 else "Not Cyberbullying"
    return result, offensive_detected

def highlight_offensive_words(input_text):
    """
    Identify offensive words or semantically similar words from the input text.
    """
    cleaned_text = clean_text(input_text)
    words = cleaned_text.split()
    detected_words = []
    
    for word in words:
        word_vector = vectorizer.transform([word]).toarray()
        for offensive_word in offensive_words:
            offensive_vector = vectorizer.transform([offensive_word]).toarray()
            similarity = cosine_similarity(word_vector, offensive_vector)[0][0]
            if similarity > 0.8:  # Adjust threshold as needed
                detected_words.append(offensive_word)
                break
                
    return detected_words

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    result, offensive_words_detected = predict_text(input_text)
    return render_template('results.html', 
                           result=result, 
                           offensive_words=offensive_words_detected, 
                           input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
