# Import required libraries for prediction app
import joblib
import uvicorn
import pandas as pd
from data_models import PredictionDataset
from pathlib import Path
from fastapi import FastAPI
from sklearn.pipeline import Pipeline


# Define application
app = FastAPI(title="NYC Taxi Trip Duration Predictor", description="NYC Taxi Trip Duration Predictor")

# Set paths
current_file_path = Path(__file__)
print()

model_path = current_file_path.parent / "models" / "models"/ "xgbreg.joblib"
preprocessor_path = model_path.parent.parent / "transformers" / "preprocessor.joblib"
output_transformer_path = preprocessor_path.parent / "output_transformer.joblib"

# Import models

model = joblib.load(model_path)
processor = joblib.load(preprocessor_path)
output_transformer = joblib.load(output_transformer_path)

# Run predict model

model_pipe = Pipeline(steps=[
    ("preprocessor", processor),
    ("regressor", model)
])

# Define homepage
@app.get("/")

def home():
    return "Welcome To NYC Taxi Trip Duration Predictor App"

@app.post("/predictions")

def do_predictions(test_data:PredictionDataset):
    X_test = pd.DataFrame(
        data = {
            'vendor_id':test_data.vendor_id,
            'passenger_count':test_data.passenger_count,
            'pickup_longitude':test_data.pickup_longitude,
            'pickup_latitude':test_data.pickup_latitude,
            'dropoff_longitude':test_data.dropoff_longitude,
            'dropoff_latitude':test_data.dropoff_latitude,
            'pickup_hour':test_data.pickup_hour,
            'pickup_date':test_data.pickup_date,
            'pickup_month':test_data.pickup_month,
            'pickup_day':test_data.pickup_day,
            'is_weekend':test_data.is_weekend,
            'haversine_distance':test_data.haversine_distance,
            'euclidean_distance':test_data.euclidean_distance,
            'manhattan_distance':test_data.manhattan_distance
        }, index=[0]
    )

    predictions = model_pipe.predict(X_test).reshape(-1,1)
    
    output_inverse_transformed = output_transformer.inverse_transform(predictions)[0].item()

    return f"Trip duration for the trip is {output_inverse_transformed:.2f} minutes"
    
    if __name__ == "__main__":
        uvicorn.run(app="app:app",
                    host="0.0.0.0",
                    port=8000)