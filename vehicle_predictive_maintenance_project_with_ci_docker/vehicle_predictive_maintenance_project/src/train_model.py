import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.data_prep import load_and_prep
import os

def train_and_save(model_path='models/rf_fault_detector.joblib'):
    X_train, X_test, y_train, y_test = load_and_prep()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Classification report:\n', classification_report(y_test, preds))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print('Model saved to', model_path)

if __name__ == '__main__':
    train_and_save()