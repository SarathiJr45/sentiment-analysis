import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from Banknote import Banknote
import numpy as np
import pickle
import pandas as pd

app= FastAPI()

pickle_in=open("model.pkl","rb")
model=pickle.load(pickle_in)

templates = Jinja2Templates(directory="Template")

@app.get("/")
def show(request:Request):
    return templates.TemplateResponse('sarathy.html',{'request': request,})


@app.post('/predict')
async def predict_Banknote(data:Banknote):
    inputs = data
    print(data)
    variance = data.Varience
    skewness = data.skewness
    curtosis = data.curtosis
    entropy = data.entropy
    prediction=model.predict([[float(variance),float(skewness),float(curtosis),float(entropy)]])
    print(prediction)
    if (prediction[0]==0):
         prediction = "Fake note"
    else:
       prediction = "Bank note"
    return {
        'prediction': prediction
    }

   

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=5000)