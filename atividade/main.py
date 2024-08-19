from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Carregando o modelo na inicialização do servidor
with open("modelo_clf.sav", "rb") as model_file:
    model = pickle.load(model_file)

# Definição do modelo de dados esperado
class InputData(BaseModel):
    input: list

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convertendo a lista para numpy array e ajustando o formato
        input_data = np.array(data.input).reshape(1, 64)  # Para uma imagem de 8x8
        
        # Fazendo a predição
        prediction = model.predict(input_data)
        
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# Valida se a API está funcionando corretamente
@app.get("/")
async def root():
    return {"message": "Conexão recebida com sucesso"}
