from fastapi import FastAPI, HTTPException
import pickle
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI()

# Tente carregar o modelo e capture possíveis exceções
model_path = "modelo_clf.sav"

if not os.path.exists(model_path):
    raise RuntimeError(f"O arquivo de modelo {model_path} não foi encontrado.")

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
        if not hasattr(model, "predict"):
            raise RuntimeError("O objeto carregado não é um modelo válido.")
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        input_data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer a previsão: {e}")
