import os
import joblib
import pandas as pd
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(
    title="Human Behavior Prediction API",
    description="Inference API for real-time human behavior classification using accelerometer data.",
    version="1.0.0"
)

# Configuration: Model and Encoder paths
# Adjust these paths if you store your joblibs in a different directory
MODEL_PATH = "xgb_model.joblib"
ENCODER_PATH = "activity_encoder.joblib"

# Global variables for loaded artifacts
model = None
encoder = None

@app.on_event("startup")
def load_artifacts():
    """Loads the model and label encoder during API startup."""
    global model, encoder
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            
        if os.path.exists(ENCODER_PATH):
            encoder = joblib.load(ENCODER_PATH)
            print(f"Encoder loaded successfully from {ENCODER_PATH}")
        else:
            print(f"Warning: Encoder file not found at {ENCODER_PATH}")
    except Exception as e:
        print(f"Error during startup artifact loading: {e}")

class SensorReading(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp or relative time in seconds")
    x_axis: float = Field(..., alias="x-axis", description="Accelerometer x-axis value")
    y_axis: float = Field(..., alias="y-axis", description="Accelerometer y-axis value")
    z_axis: float = Field(..., alias="z-axis", description="Accelerometer z-axis value")

    class Config:
        allow_population_by_field_name = True

class InferenceRequest(BaseModel):
    readings: List[SensorReading] = Field(..., description="A sequence of exactly 50 consecutive sensor readings.")

def extract_features_api(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the exact training feature engineering logic.
    Expects input dataframe with columns: ['timestamp', 'x-axis', 'y-axis', 'z-axis']
    Returns a single row (the 50th sample) with all calculated features.
    """
    # 1. Magnitude of acceleration
    df_features = df_input.copy()
    df_features["sq_acc"] = df_features["x-axis"]**2 + df_features["y-axis"]**2 + df_features["z-axis"]**2

    axes = ['x-axis', 'y-axis', 'z-axis', 'sq_acc']
    window_size = 50

    for axis in axes:
        # 2. Rolling Stats
        df_features[f'{axis}_mean'] = df_features[axis].rolling(window=window_size).mean()
        df_features[f'{axis}_std'] = df_features[axis].rolling(window=window_size).std()

        # 3. Signal Jerk
        df_features[f'{axis}_jerk'] = df_features[axis].diff().fillna(0)

        # 4. EMA for smoothing (Span=3)
        if axis != 'sq_acc':
            df_features[f'{axis}_ema'] = df_features[axis].ewm(span=3).mean()

    # 5. Axis Correlations
    df_features['xy_corr'] = df_features['x-axis'].rolling(window=window_size).corr(df_features['y-axis'])
    df_features['yz_corr'] = df_features['y-axis'].rolling(window=window_size).corr(df_features['z-axis'])
    df_features['xz_corr'] = df_features['x-axis'].rolling(window=window_size).corr(df_features['z-axis'])

    # Final feature column selection (matching ml_pipeline.py)
    feature_cols = [
        "timestamp", "x-axis", "y-axis", "z-axis", "sq_acc",
        "x-axis_std", "x-axis_mean", "y-axis_std", "y-axis_mean",
        "z-axis_std", "z-axis_mean", "sq_acc_mean", "sq_acc_std",
        "x-axis_ema", "y-axis_ema", "z-axis_ema",
        "x-axis_jerk", "y-axis_jerk", "z-axis_jerk", "sq_acc_jerk",
        "xy_corr", "yz_corr", "xz_corr"
    ]

    # Return only the last row (the one that has full 50-window info)
    return df_features[feature_cols].tail(1)

@app.get("/")
def read_root():
    return {"message": "Human Behavior Prediction API is online. Send 50 samples to /predict."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None
    }

@app.post("/predict")
async def predict(request: InferenceRequest):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Check server startup logs.")

    if len(request.readings) < 50:
        raise HTTPException(
            status_code=400, 
            detail="Insufficient data. The model requires exactly 50 consecutive readings to calculate windowed features."
        )

    # Convert Pydantic models to Pandas DataFrame
    data = [reading.dict(by_alias=True) for reading in request.readings]
    df_input = pd.DataFrame(data)

    # Pre-sort by timestamp just in case
    df_input = df_input.sort_values("timestamp")

    try:
        # Extract features (Window logic)
        X_inference = extract_features_api(df_input)
        
        # Clean potential NaNs from the very first diff or edge cases
        X_inference = np.nan_to_num(X_inference)

        # Prediction
        prediction_num = model.predict(X_inference)[0]
        prediction_label = encoder.inverse_transform([prediction_num])[0]

        return {
            "activity_id": int(prediction_num),
            "activity": str(prediction_label),
            "samples_processed": len(request.readings)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # If running locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
