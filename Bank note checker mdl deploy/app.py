import uvicorn
from fastapi import FastAPI
from Banknote import Banknote
import numpy as np
import pickle
import pandas as pd

app= FastAPI()
pickle_in=open("model.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
def index():
    return {'Message':'Hello world'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}

@app.post('/predict')
def predict_Banknote(data:Banknote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    
    if (prediction[0]>0.5):
        print('Fake Note')
    else:
        print('Bank Note')
    return {
        'prediction': str(prediction)
        }

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)