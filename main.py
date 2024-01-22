from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy.sparse import hstack
from sklearn.svm import SVC
import joblib
import numpy as np
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="Template")
class InputData(BaseModel):
    text: str

# Load the pre-trained model, vectorizer, and Word2Vec model
svm_model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
word2vec_model = joblib.load('word2vec_model.joblib')

def avg_word_vector(text, model):
    word_vectors = [model.wv[word] for word in text if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Mapping between numeric labels and text labels
label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

@app.get("/")
def show(request:Request):
    return templates.TemplateResponse('index.html',{'request': request,})


@app.post("/predict/")
async def predict_sentiment(data: InputData):
    cleaned_input = ''.join(char for char in data.text if char.isalnum() or char.isspace())
    cleaned_input = cleaned_input.lower()

    input_vectorized = vectorizer.transform([cleaned_input])

    input_word2vec = avg_word_vector(cleaned_input.split(), word2vec_model)

    input_word2vec_normalized = np.array(input_word2vec).reshape(1, -1)

    input_combined = hstack((input_vectorized, input_word2vec_normalized))

    prediction = svm_model.predict(input_combined)[0]

    predicted_sentiment = label_mapping[prediction]

    return JSONResponse(content={"predicted_sentiment": predicted_sentiment}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
