import numpy as np
import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI
from heart_data import Heart
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler
import joblib

app = FastAPI()
#classifier
sc = joblib.load('scaler.save')
classifier = joblib.load('finalized_model.sav')

@app.post('/predict_heart')
def predict_heart(data:Heart):
    data = data.dict()
    age = data['age']
    sex = data['sex']
    cp = data['cp']
    trestbps = data['trestbps']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    exang = data['exang']
    oldpeak = data['oldpeak']
    slope = data['slope']
    ca = data['ca']
    thal = data['thal']
    data = sc.transform([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    
    prediction = classifier.predict(data)
    
    if(prediction[0]<=0.5):
        prediction="NO chances of getting heart disease"
    else:
        prediction="YOU have Chances of getting heart disease"
    return {'prediction': prediction}


