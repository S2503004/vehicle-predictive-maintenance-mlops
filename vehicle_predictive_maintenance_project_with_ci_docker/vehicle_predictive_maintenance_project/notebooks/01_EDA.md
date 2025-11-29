# 01 — Exploratory Data Analysis (EDA)

Open this file for a guided EDA using the synthetic dataset `data/vehicle_sensor_data_synthetic.csv`.

Suggested steps:
1. Load dataset with pandas.
2. Check basic stats: `df.describe()`, `df.info()`.
3. Visualize distributions: speed, engine_temp, vibration.
4. Check correlation matrix between numeric features.
5. Compare sensor stats grouped by `fault` value.
6. Plot time-to-failure distribution (where available).
7. Save figures to `reports/figures/` if needed.

Example code snippets:

```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/vehicle_sensor_data_synthetic.csv')
print(df.describe())
df['engine_temp_c'].hist()
plt.show()
```