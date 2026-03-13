
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(
    title="Agricultural Recommendation API",
    description="API for predicting optimal crop planting times and recommending crops based on conditions",
    version="1.0.0"
)

# Allow requests from frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Load Datasets =====================
try:
    crop_data = pd.read_csv("../data/crop.csv")
    rainfall_data = pd.read_csv("../data/rainfall.csv")
    temperature_data = pd.read_csv("../data/temperature.csv")
    
    # Convert state and district names to lowercase for consistency
    rainfall_data['STATE/UT'] = rainfall_data['STATE/UT'].str.lower().str.strip()
    rainfall_data['DISTRICT'] = rainfall_data['DISTRICT'].str.lower().str.strip()
except Exception as e:
    print(f"Error loading datasets: {e}")
    # The app will still start, but endpoints will fail if files aren't found

# ===================== Temperature Prediction =====================
temperature_data.columns = temperature_data.columns.str.strip()
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
month_mapping = {month.lower(): idx + 1 for idx, month in enumerate(months)}
month_idx_to_name = {idx + 1: month for idx, month in enumerate(months)}

temperature_melted = pd.melt(temperature_data, id_vars='YEAR', value_vars=months, var_name='Month', value_name='Temperature')
temperature_melted['Month'] = temperature_melted['Month'].str.lower().map(month_mapping)

X_temp = temperature_melted[['YEAR', 'Month']]
y_temp = temperature_melted['Temperature']
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

temperature_model = GradientBoostingRegressor(n_estimators=300, random_state=42)
temperature_model.fit(X_train_temp, y_train_temp)

# ===================== Rainfall Prediction =====================
rainfall_columns = ['STATE/UT', 'DISTRICT'] + months
rainfall_data = rainfall_data[rainfall_columns]
rainfall_data.fillna(rainfall_data.mean(numeric_only=True), inplace=True)

rainfall_long = pd.melt(rainfall_data, id_vars=['STATE/UT', 'DISTRICT'], value_vars=months, var_name='Month', value_name='Rainfall')
rainfall_long['Month'] = rainfall_long['Month'].str.lower().map(month_mapping)

label_encoder_state = LabelEncoder()
label_encoder_district = LabelEncoder()

rainfall_long['STATE/UT_encoded'] = label_encoder_state.fit_transform(rainfall_long['STATE/UT'])
rainfall_long['DISTRICT_encoded'] = label_encoder_district.fit_transform(rainfall_long['DISTRICT'])

X_rainfall = rainfall_long[['STATE/UT_encoded', 'DISTRICT_encoded', 'Month']]
y_rainfall = rainfall_long['Rainfall']
X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(X_rainfall, y_rainfall, test_size=0.2, random_state=42)

rainfall_model = GradientBoostingRegressor(n_estimators=300, random_state=42)
rainfall_model.fit(X_train_rain, y_train_rain)

# Create mappings for lookup
state_mapping = dict(zip(rainfall_long['STATE/UT'].unique(), label_encoder_state.transform(rainfall_long['STATE/UT'].unique())))
district_mapping = dict(zip(rainfall_long['DISTRICT'].unique(), label_encoder_district.transform(rainfall_long['DISTRICT'].unique())))

# ===================== Crop Recommendation =====================
X_crop = crop_data.drop(["label", "humidity"], axis=1)
y_crop = crop_data["label"]
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42, stratify=y_crop)

rf_crop_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_crop_model.fit(X_train_crop, y_train_crop)

# ===================== Soil Requirements Prediction =====================
soil_features = ["N", "P", "K", "ph"]
optimal_conditions = crop_data.groupby("label")[soil_features].mean()

# ===================== API Request Models =====================
class BestTimeRequest(BaseModel):
    crop_name: str
    year: int
    state: str
    district: str

class CropRecommendRequest(BaseModel):
    state: str
    district: str
    year: int
    month: str
    N: float
    P: float
    K: float
    pH: float

# ===================== Helper Functions =====================
def get_weather_predictions(state: str, district: str, year: int, month: str):
    if month.upper() not in [m for m in months]:
        raise HTTPException(status_code=400, detail="Invalid month. Use format: JAN, FEB, etc.")
    
    try:
        state_lower = state.lower()
        district_lower = district.lower()
        
        if state_lower not in state_mapping:
            raise HTTPException(status_code=400, detail=f"State '{state}' not found in dataset")
            
        if district_lower not in district_mapping:
            raise HTTPException(status_code=400, detail=f"District '{district}' not found in dataset")
            
        state_encoded = state_mapping[state_lower]
        district_encoded = district_mapping[district_lower]
        month_encoded = month_mapping[month.lower()]

        temperature_pred = temperature_model.predict([[year, month_encoded]])[0]
        rainfall_pred = rainfall_model.predict([[state_encoded, district_encoded, month_encoded]])[0]

        return round(temperature_pred, 2), round(rainfall_pred, 2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

# ===================== API Endpoints =====================
@app.get("/")
async def root():
    return {"message": "Welcome to the Agricultural Recommendation API"}

@app.post("/predict_best_time/")
def predict_best_time(request: BestTimeRequest):
    """
    Predict the best time (month) to plant a specific crop based on weather and soil conditions
    """
    crop_name = request.crop_name.lower().strip()
    year = request.year
    state = request.state.lower().strip()
    district = request.district.lower().strip()
    
    # Check if crop exists in the dataset
    if crop_name not in optimal_conditions.index:
        raise HTTPException(status_code=400, detail=f"Crop '{crop_name}' not found in the dataset.")
    
    # Get average soil requirements for the crop
    soil_requirements = optimal_conditions.loc[crop_name].to_dict()
    
    try:
        # Validate state and district
        if state not in state_mapping:
            raise HTTPException(status_code=400, detail=f"State '{state}' not found in dataset")
        if district not in district_mapping:
            raise HTTPException(status_code=400, detail=f"District '{district}' not found in dataset")
            
        state_encoded = state_mapping[state]
        district_encoded = district_mapping[district]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing location: {str(e)}")
    
    best_month = None
    best_proba = -1
    monthly_predictions = []
    
    # Evaluate each month
    for month_num in range(1, 13):
        # Predict weather for the month
        temp_pred = temperature_model.predict([[year, month_num]])[0]
        rainfall_pred = rainfall_model.predict([[state_encoded, district_encoded, month_num]])[0]
        
        # Create input for the crop recommendation model
        crop_input = pd.DataFrame([{
            "N": soil_requirements["N"],
            "P": soil_requirements["P"],
            "K": soil_requirements["K"],
            "temperature": temp_pred,
            "ph": soil_requirements["ph"],
            "rainfall": rainfall_pred
        }], columns=X_crop.columns)
        
        # Get the probability of the crop being recommended
        proba = rf_crop_model.predict_proba(crop_input)[0]
        crop_index = list(rf_crop_model.classes_).index(crop_name)
        probability = proba[crop_index]
        
        # Update best month if this probability is higher
        if probability > best_proba:
            best_proba = probability
            best_month = months[month_num - 1]
        
        # Store predictions for the response
        monthly_predictions.append({
            "month": months[month_num - 1],
            "rainfall": round(rainfall_pred, 2),
            "temperature": round(temp_pred, 2),
            "probability": round(probability, 4)
        })
    
    # Return the result
    return {
        "crop_name": crop_name,
        "year": year,
        "state": state,
        "district": district,
        "soil_requirements": {k: round(v, 2) for k, v in soil_requirements.items()},
        "best_month": best_month,
        "monthly_predictions": monthly_predictions
    }

@app.post("/recommend_crop/")
def recommend_crop(request: CropRecommendRequest):
    """
    Recommend a crop based on soil conditions and predicted weather
    """
    temperature_pred, rainfall_pred = get_weather_predictions(
        request.state, 
        request.district, 
        request.year, 
        request.month
    )
    
    # Create input DataFrame with only the features used during training (without humidity)
    crop_input = pd.DataFrame([{
        "N": request.N, 
        "P": request.P, 
        "K": request.K, 
        "temperature": temperature_pred, 
        "ph": request.pH, 
        "rainfall": rainfall_pred
    }], columns=X_crop.columns)
    
    predicted_crop = rf_crop_model.predict(crop_input)[0]

    return {
        "year": request.year,
        "month": request.month.upper(),
        "state": request.state,
        "district": request.district,
        "soil_conditions": {
            "N": request.N,
            "P": request.P,
            "K": request.K,
            "pH": request.pH
        },
        "predicted_temperature": temperature_pred,
        "predicted_rainfall": rainfall_pred,
        "recommended_crop": predicted_crop
    }

# For running the application directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)