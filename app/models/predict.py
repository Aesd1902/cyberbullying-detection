import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from text_preprocessor import clean_text, vectorize_text
from sklearn.metrics.pairwise import cosine_similarity  # <-- Import the cosine_similarity function
import os

# Paths for saving model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current folder
MODEL_PATH = os.path.join(BASE_DIR, 'cyberbullying_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
DATASET_PATH = os.path.join(BASE_DIR, '../../data/raw/cyberbullying_data.csv')  # Adjust path as needed

# Load dataset
data = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1')  # Use appropriate encoding if necessary
data['cleaned_text'] = data['text'].apply(clean_text)

# Vectorize the text data
X, vectorizer = vectorize_text(data['cleaned_text'])
y = data['label']  # Labels (0: Not Cyberbullying, 1: Cyberbullying)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test data: {accuracy * 100:.2f}%')

def find_most_similar_text(input_text, label):
    """
    Find the most similar text from the dataset with the specified label.
    """
    # Filter dataset based on label
    filtered_data = data[data['label'] == label]

    # Calculate cosine similarity
    vectorized_input = vectorizer.transform([input_text])
    vectorized_texts = vectorizer.transform(filtered_data['cleaned_text'])
    similarities = cosine_similarity(vectorized_input, vectorized_texts)

    # Get the most similar text
    if similarities[0].size > 0:
        max_index = similarities[0].argmax()
        most_similar_text = filtered_data['text'].iloc[max_index]
        return most_similar_text
    return "No similar text found."

def predict_text(input_text):
    """
    Predict cyberbullying content and find the most similar text from the dataset.
    """
    cleaned_text = clean_text(input_text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)

    if prediction[0] == 1:  # Cyberbullying
        result = "Cyberbullying"
        similar_text = find_most_similar_text(cleaned_text, label=1)
    else:
        result = "Not Cyberbullying"
        similar_text = find_most_similar_text(cleaned_text, label=0)
    
    return result, similar_text

if __name__ == "__main__":
    # To test and train the model
    print("Training completed and model saved.")
    
    # Now you can run prediction for an input text:
    test_text = input("Enter text to analyze: ")
    prediction, similar_text = predict_text(test_text)
    print(f"Input: {test_text}")
    print(f"Prediction: {prediction}")
    print(f"Similar Text: {similar_text}")
