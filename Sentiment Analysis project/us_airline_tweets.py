# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('C:\\US airline Tweets model\\Tweets.csv')

df['airline_sentiment'] = df['airline_sentiment'].str.lower()
df['text'] = df['text'].str.lower()

def clean_text(text):
    return ''.join(char for char in text if char.isalnum() or char.isspace())

# Apply the function to the 'Text' column
df['airline_sentiment'] = df['airline_sentiment'].apply(clean_text)
df['text'] = df['text'].apply(clean_text)

y=df['airline_sentiment']
x=df['text']

# # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
le= LabelEncoder()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

#Word2vec
from gensim.models import Word2Vec
import numpy as np
tokenized_text= [text.split() for text in df['text']]
word2vec_model=Word2Vec(tokenized_text,vector_size=100,min_count=1,window=5,sg=0)

def avg_word_vector(text, model):
    word_vectors = [model.wv[word] for word in text if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

X_train_word2vec = [avg_word_vector(text.split(), word2vec_model) for text in X_train]
X_test_word2vec = [avg_word_vector(text.split(), word2vec_model) for text in X_test]

from scipy.sparse import hstack
from sklearn.svm import SVC

# Concatenate TF-IDF features and Word2Vec features
X_train_combined = hstack((X_train_vectorized, np.array(X_train_word2vec)))
X_test_combined = hstack((X_test_vectorized, np.array(X_test_word2vec)))

# Train a Linear SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_combined, y_train)
y_pred_svm = svm_model.predict(X_test_combined)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(accuracy_svm)

import joblib
joblib.dump(svm_model,'model.joblib')
joblib.dump(vectorizer,'vectorizer.joblib')
joblib.dump(word2vec_model,'word2vec_model.joblib')
