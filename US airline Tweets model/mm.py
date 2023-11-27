from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from joblib import load
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from fastapi.responses import JSONResponse

# Load the trained model and preprocessing steps
svm_model = load('model.joblib')  # Replace with your actual model file
vectorizer = load('vectorizer.joblib')  # Replace with your actual vectorizer file
word2vec_model = load('word2vec_model.joblib')  # Replace with your actual Word2Vec model file
le= LabelEncoder()

app = FastAPI()

# Define input data schema using Pydantic BaseModel
class InputData(BaseModel):
    text: str

label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


@app.post("/predict/")
async def predict_sentiment(data: InputData):
    # Clean and preprocess the input
    cleaned_input = ''.join(char for char in data.text if char.isalnum() or char.isspace())
    cleaned_input = cleaned_input.lower()
    # Generate Word2Vec embeddings
    def avg_word_vector(text, model):
        word_vectors = [model.wv[word] for word in text if word in model.wv]
        if not word_vectors:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)
    input_text = cleaned_input
    

    # Assuming 'new_input' is the cleaned and processed input text
    new_input_vectorized = vectorizer.transform([cleaned_input])
    input_word2vec = avg_word_vector(cleaned_input.split(), word2vec_model)
    input_word2vec_normalized = np.array(input_word2vec).reshape(1, -1)

    # Concatenate TF-IDF features and Word2Vec features for the new input
    new_input_combined = hstack((new_input_vectorized, np.array(input_word2vec_normalized)))

    # Predict sentiment for the new input
    prediction = svm_model.predict(new_input_combined)

    # Decode the predicted label if needed (if you used LabelEncoder)
    predicted_sentiment = le.fit_transform(prediction)[0]
    predicted_sentiment = label_mapping[predicted_sentiment]

    return JSONResponse(content={"predicted_sentiment": predicted_sentiment}, status_code=200)



if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)