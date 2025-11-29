import streamlit as st
import pandas as pd
import joblib
from src.data_prep import load_and_prep

st.set_page_config(page_title='Vehicle Predictive Maintenance (Demo)')

st.title('Vehicle Predictive Maintenance — Demo')
st.markdown('Upload sensor CSV or use synthetic sample. The app runs a pre-trained classifier to flag likely faults.')

uploaded = st.file_uploader('Upload CSV with sensor columns', type=['csv'])
model_path = 'models/rf_fault_detector.joblib'

if uploaded is None:
    st.info('Using sample from `data/vehicle_sensor_data_synthetic.csv`')
    df = pd.read_csv('data/vehicle_sensor_data_synthetic.csv').head(200)
else:
    df = pd.read_csv(uploaded)

st.dataframe(df.head(), use_container_width=True)

if st.button('Run predictions (demo)'):
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        st.error('Model not found — run `python src/train_model.py` first to train and save the model.')
        st.stop()
    features = ['speed_kmph','engine_temp_c','oil_pressure_psi','vibration_g','mileage_km','age_months']
    X = df[features].fillna(0)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:,1]
    out = df[['vehicle_id','timestamp']].copy()
    out['fault_pred'] = preds
    out['fault_prob'] = probs
    st.success('Predictions complete')
    st.dataframe(out.head(50), use_container_width=True)