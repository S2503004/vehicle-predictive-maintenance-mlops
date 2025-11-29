import joblib
import pandas as pd
import os

MODEL_PATH = 'models/rf_fault_detector.joblib'

def predict_from_row(row_dict):
    clf = joblib.load(MODEL_PATH)
    df = pd.DataFrame([row_dict])
    preds = clf.predict(df)
    probs = clf.predict_proba(df)[:,1]
    return int(preds[0]), float(probs[0])

if __name__ == '__main__':
    sample = {
        'speed_kmph': 80,
        'engine_temp_c': 105,
        'oil_pressure_psi': 30,
        'vibration_g': 0.6,
        'mileage_km': 120000,
        'age_months': 60
    }
    print('Predict (fault, prob):', predict_from_row(sample))