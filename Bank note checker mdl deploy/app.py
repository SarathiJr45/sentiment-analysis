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
def index():
    return {'Message':'Hello world'}

# @app.get('/{name}')
# def get_name(name: str):
#     return {'Welcome': f'{name}'}

@app.get('/predict/{data}', response_class=HTMLResponse)
async def predict_Banknote(request: Request, data:str):
    # data = data.dict()
    data= data.split(",")
    variance=float(data[0])
    skewness=float(data[1])
    curtosis=float(data[2])
    entropy=float(data[3])
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    
    # if (prediction[0]>0.5):
    #     return {'prediction': "Fake note"}
    #     # print('Fake Note')
    # else:
    #     return {'prediction': "Bank note"}
    #     # print('Bank Note')
    return templates.TemplateResponse("index.html", {"request":request, "output": str(prediction)})

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)