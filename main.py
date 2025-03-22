import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS (Enable API access from any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = "lstmfinalmodel.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

# Define API input and output models using Pydantic
class PredictionRequest(BaseModel):
    tickers: List[str]  # List of stock tickers
    days: int           # Number of future days to predict

class StockPredictionResponse(BaseModel):
    stock: str                        # Stock ticker
    closing_prices: Dict[str, float]   # Historical closing prices {date: price}
    test_predictions: List[float]      # Test predictions
    test_dates: List[str]              # Test dates
    future_predictions: List[float]    # Future predictions
    future_dates: List[str]            # Future dates

class PredictionResponse(BaseModel):
    results: List[StockPredictionResponse]  # List of predictions for each stock

# Define API endpoints
@app.get("/")
def root():
    return {"message": "Stock Price Prediction API is running!"}

# Read the port from the environment (default: 10000)
PORT = int(os.getenv("PORT", 10000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
