from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

class Verify(BaseModel):
    x1 : float
    x2 : float
app = FastAPI()

@app.get("/status")
def health():
    return {"status": "good"}
@app.get("/about")
def about():
    return {
      "project": "Concentric Circles Classifier",
      "framework": "PyTorch"
    }
@app.post("/predict")
def predict_endpoint(point :Verify):
    return predict(point.x1, point.x2) #we imported the predict function which came from the file that imported the model so we can just run everything here on the api using other files
    
            
