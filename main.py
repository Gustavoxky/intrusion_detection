from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import xgboost as xgb
import cupy as cp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb_model = joblib.load(os.path.join(BASE_DIR, "model/xgb_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model/scaler.pkl"))

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    try:
        if "features" not in data:
            raise HTTPException(status_code=400, detail="Faltando 'features' no corpo da solicitação.")
        
        features = np.array([data["features"]], dtype=np.float32)
        
        expected_features = scaler.n_features_in_
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Número incorreto de features: {features.shape[1]}. Esperado: {expected_features}."
            )
        
        features_scaled = scaler.transform(features)
        
        features_scaled_gpu = cp.array(features_scaled)
        
        prediction_gpu = xgb_model.predict(features_scaled_gpu)
        
        prediction = cp.asnumpy(prediction_gpu)
        
        result = "Normal" if prediction[0] == 0 else "Intrusão"
        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
