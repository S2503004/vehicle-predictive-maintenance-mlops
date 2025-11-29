# Vehicle Predictive Maintenance — End-to-End Data Analytics Project

**What it is:** An end-to-end analytics pipeline for fleet vehicle predictive maintenance:
- Synthetic vehicle sensor dataset (speed, engine temp, oil pressure, vibration, mileage)
- EDA notebook
- Model training (classification for fault detection + regression for time-to-failure)
- Simple Streamlit dashboard demo
- Instructions to run locally and on GitHub

**Files & structure**
```
/vehicle_predictive_maintenance_project
├─ data/
│  └─ vehicle_sensor_data_synthetic.csv
├─ notebooks/
│  └─ 01_EDA.md
├─ src/
│  ├─ data_prep.py
│  ├─ train_model.py
│  └─ predict.py
├─ app/
│  └─ dashboard_app.py
├─ requirements.txt
├─ LICENSE
└─ README.md
```

**How to run (quick)**
1. Clone or unzip the repo.
2. Create a Python environment (Python 3.9+ recommended).
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run EDA (open `notebooks/01_EDA.md` or run scripts).
5. Train models:
```bash
python src/train_model.py
```
6. Launch dashboard (demo):
```bash
streamlit run app/dashboard_app.py
```

**Notes**
- Data is synthetic and meant for learning and demo. Replace `data/vehicle_sensor_data_synthetic.csv` with real sensor logs for production.
- Model artifacts will be saved in `models/` after training.

**License:** MIT

---

## Continuous Integration (GitHub Actions)

A CI workflow is included at `.github/workflows/ci.yml`. It runs on push and pull request events to `main`/`master`. The workflow installs dependencies and runs basic smoke checks. The workflow is intentionally lightweight for PRs; it trains a model only on push events. Customize the workflow to suit your repository policies (e.g., add unit tests, linting, or faster mock tests).

## Docker (demo)

A demo `Dockerfile` and `.dockerignore` are included to build a container for the Streamlit demo app.

Build and run locally:
```bash
docker build -t vehicle-pm-demo:latest .
docker run -p 8501:8501 vehicle-pm-demo:latest
```

Notes:
- The Docker image is intended for demo/testing. For production, add multi-stage builds, smaller base images, non-root user, and pinned dependency versions.
- Excluding `data/` and `models/` from the image keeps the container light. Mount real datasets and models as volumes at runtime.