from fastapi import FastAPI, Query, HTTPException
from datetime import datetime, timedelta
import os, json, joblib, numpy as np

app = FastAPI(title="Open Meteo – AMLA AT2", version="1.0.0")

# Paths
BASE = os.path.dirname(os.path.dirname(__file__))
RDIR = os.path.join(BASE, "models", "rain_or_not")
PDR = os.path.join(BASE, "models", "precipitation_fall")

def _load(path):
    return joblib.load(path) if os.path.exists(path) else None

# --- Rain Model (Classification) ---
clf = _load(os.path.join(RDIR, "model.joblib"))
clf_pre = _load(os.path.join(RDIR, "preprocessor.joblib"))

clf_thresh = 0.5
tpath = os.path.join(RDIR, "threshold.json")
if os.path.exists(tpath):
    with open(tpath) as f:
        clf_thresh = json.load(f).get("threshold", 0.5)

clf_features = []
fpath = os.path.join(RDIR, "feature_list.json")
if os.path.exists(fpath):
    with open(fpath) as f:
        clf_features = json.load(f)

# --- Precipitation Model (Regression) ---
reg = _load(os.path.join(PDR, "ridge_model_nb2.joblib"))
reg_scaler = _load(os.path.join(PDR, "standard_scaler.pkl"))

reg_features = []
fpath = os.path.join(PDR, "feature_list.json")
if os.path.exists(fpath):
    with open(fpath) as f:
        reg_features = json.load(f)

# --- Helpers ---
def parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, "Invalid date format. Use YYYY-MM-DD.")

def build_feature_vector(features: list[str], preprocessor) -> np.ndarray:
    """
    Builds a dummy zero vector aligned with the expected feature count.
    Uses preprocessor.n_features_in_ if available.
    """
    if hasattr(preprocessor, "n_features_in_"):
        n = preprocessor.n_features_in_
    else:
        n = len(features)
    return np.zeros((1, n))

# --- Endpoints ---
@app.get("/")
def root():
    return {
        "project": "Open Meteo – AMLA AT2",
        "description": "Weather forecast ML models served via FastAPI",
        "endpoints": {
            "/health/": "GET",
            "/predict/rain/": "GET?date=YYYY-MM-DD",
            "/predict/precipitation/fall/": "GET?date=YYYY-MM-DD"
        },
        "expected_inputs": {
            "date": "string (YYYY-MM-DD)"
        },
        "github_repo": "https://github.com/Mitesh2412/AMLA_AT2_API"
    }

@app.get("/health/")
def health():
    return {"status": "ok", "msg": "Welcome to Open Meteo AMLA AT2 API"}

@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., description="YYYY-MM-DD")):
    d0 = parse_date(date)
    d7 = d0 + timedelta(days=7)

    if clf is None or clf_pre is None:
        raise HTTPException(503, "Rain model not loaded.")

    # Build aligned feature vector
    X = build_feature_vector(clf_features, clf_pre)
    Xp = clf_pre.transform(X)

    if hasattr(clf, "predict_proba"):
        proba = float(clf.predict_proba(Xp)[0, 1])
    else:
        proba = float(clf.predict(Xp)[0])

    return {
        "input_date": d0.strftime("%Y-%m-%d"),
        "prediction": {
            "date": d7.strftime("%Y-%m-%d"),
            "will_rain": bool(proba >= clf_thresh),
            "probability": round(proba, 3)
        }
    }

@app.get("/predict/precipitation/fall/")
def predict_precip(date: str = Query(..., description="YYYY-MM-DD")):
    d0 = parse_date(date)
    d1 = d0 + timedelta(days=1)
    d3 = d0 + timedelta(days=3)

    if reg is None or reg_scaler is None:
        raise HTTPException(503, "Precipitation model not loaded.")

    # Build aligned feature vector
    X = build_feature_vector(reg_features, reg_scaler)
    Xs = reg_scaler.transform(X)

    y = float(reg.predict(Xs)[0])

    return {
        "input_date": d0.strftime("%Y-%m-%d"),
        "prediction": {
            "start_date": d1.strftime("%Y-%m-%d"),
            "end_date": d3.strftime("%Y-%m-%d"),
            "precipitation_fall": f"{y:.1f}"
        }
    }
