# Weather Prediction API 🌦️

A FastAPI-based REST API serving two machine learning models trained on Sydney weather data from [Open-Meteo](https://open-meteo.com/). Built as part of a Machine Learning Engineering project at the University of Technology Sydney.

## What It Does

| Endpoint | Task | Model | Performance |
|---|---|---|---|
| `/predict/rain/` | Will it rain 7 days from a given date? | Random Forest Classifier | ROC-AUC: 0.583, Accuracy: 0.61 |
| `/predict/precipitation/fall/` | Total precipitation (mm) over next 3 days | Ridge Regression | R²: 0.777, RMSE: 6.15mm |

## Tech Stack

- **API:** FastAPI + Uvicorn
- **ML:** scikit-learn, joblib
- **Models:** Random Forest Classifier, Ridge Regression
- **Deployment:** Docker
- **Data Source:** Open-Meteo API (Sydney, Australia)

## Project Structure

```
weather-prediction-api/
├── app/
│   └── main.py              # FastAPI app with all endpoints
├── models/
│   ├── rain_or_not/         # Classification model + preprocessor + metrics
│   └── precipitation_fall/  # Regression model + scaler + metrics
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Run locally

```bash
# Clone the repo
git clone https://github.com/Mitesh2412/weather-prediction-api.git
cd weather-prediction-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`

### Run with Docker

```bash
docker build -t weather-api .
docker run -p 8000:8000 weather-api
```

## API Endpoints

### `GET /`
Returns project info and available endpoints.

### `GET /health/`
Health check.

### `GET /predict/rain/?date=YYYY-MM-DD`
Predicts whether it will rain exactly 7 days from the given date.

**Example:**
```bash
curl "http://localhost:8000/predict/rain/?date=2025-10-01"
```

**Response:**
```json
{
  "input_date": "2025-10-01",
  "prediction": {
    "date": "2025-10-08",
    "will_rain": false,
    "probability": 0.241
  }
}
```

### `GET /predict/precipitation/fall/?date=YYYY-MM-DD`
Predicts total precipitation (mm) over the 3-day window starting the day after the input date.

**Example:**
```bash
curl "http://localhost:8000/predict/precipitation/fall/?date=2025-10-01"
```

**Response:**
```json
{
  "input_date": "2025-10-01",
  "prediction": {
    "start_date": "2025-10-02",
    "end_date": "2025-10-04",
    "precipitation_fall": "3.2"
  }
}
```

## Model Details

### Rain Classification (Random Forest)
- **Target:** Binary — will it rain in exactly +7 days?
- **Threshold:** 0.3 (optimised for recall — better to predict rain when unsure)
- **Test ROC-AUC:** 0.583
- **Test Accuracy:** 60.6%

### Precipitation Regression (Ridge)
- **Target:** Total precipitation (mm) over next 3 days
- **Features:** Lag features (1, 3, 7 days), rolling averages (3, 7 days), humidity, temperature, sunshine duration, wind direction, cyclical month encoding
- **Test R²:** 0.777
- **Test RMSE:** 6.15mm

## Notes

This project was developed as part of the Applied Machine Learning & AI (AMLA) subject in the Master of Data Science & Innovation program at UTS Sydney. The models are trained on historical Sydney weather data fetched from the Open-Meteo API.

---

*Built with FastAPI · scikit-learn · Docker*
