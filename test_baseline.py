import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

INPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\processed'
OUTPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\results\baselines'

FEATURE_COLS = ['NTL', 'TEM', 'SP', 'PRE', 'ET', 'UW', 'VW', 'DSR', 'AOD', 'NDVI', 'CO', 'NO2', 'SO2', 'O3', 'DEM']
TARGET_COL = 'XCO2'

print("Loading data...")
all_data = []
for year in range(2017, 2021):
    for month in range(1, 13):
        filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
        filepath = os.path.join(INPUT_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(df)} samples")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("\nTraining Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
print(f"RF - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
result_df = pd.DataFrame({
    'predicted': preds,
    'target': y_test.values
})
result_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_predictions.csv'), index=False)
print(f"\nResults saved to {OUTPUT_DIR}")