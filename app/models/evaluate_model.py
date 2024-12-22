import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from text_preprocessor import clean_text

# Load dataset
data = pd.read_csv('data/raw/cyberbullying_data.csv')
data['cleaned_text'] = data['text'].apply(clean_text)

# Load vectorizer and transform data
vectorizer = joblib.load('app/models/vectorizer.pkl')
X = vectorizer.transform(data['cleaned_text']).toarray()
y = data['label']

# Load model
model = joblib.load('app/models/cyberbullying_model.pkl')

# Evaluate model
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
