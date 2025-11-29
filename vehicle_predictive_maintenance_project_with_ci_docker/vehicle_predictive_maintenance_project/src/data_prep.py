import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_prep(path='data/vehicle_sensor_data_synthetic.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    # basic cleansing
    df = df.dropna(subset=['speed_kmph','engine_temp_c','oil_pressure_psi','vibration_g'])
    # features and targets for classification (fault)
    X = df[['speed_kmph','engine_temp_c','oil_pressure_psi','vibration_g','mileage_km','age_months']]
    y = df['fault']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prep()
    print('Shapes:', X_train.shape, X_test.shape)