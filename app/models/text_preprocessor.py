import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Clean text by removing URLs, punctuation, and stopwords."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def vectorize_text(data, max_features=1000):
    """Convert text into numerical format using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform(data).toarray(), vectorizer
