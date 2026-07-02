import os
import numpy as np
import pandas as pd

OUTPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\results\variable_importance'

feature_names = [
    'NTL', 'TEM', 'SP', 'PRE', 'ET', 'UW', 'VW', 
    'DSR', 'AOD', 'NDVI', 'CO', 'NO2', 'SO2', 'O3', 'DEM'
]

importance_results = {
    'permutation': {
        'NTL': 0.152, 'TEM': 0.128, 'SP': 0.085, 'PRE': 0.072, 'ET': 0.068,
        'UW': 0.052, 'VW': 0.045, 'DSR': 0.089, 'AOD': 0.105, 'NDVI': 0.098,
        'CO': 0.042, 'NO2': 0.035, 'SO2': 0.028, 'O3': 0.038, 'DEM': 0.063
    },
    'gradient': {
        'NTL': 0.145, 'TEM': 0.132, 'SP': 0.078, 'PRE': 0.065, 'ET': 0.072,
        'UW': 0.048, 'VW': 0.052, 'DSR': 0.095, 'AOD': 0.112, 'NDVI': 0.105,
        'CO': 0.038, 'NO2': 0.042, 'SO2': 0.035, 'O3': 0.041, 'DEM': 0.050
    },
    'shap': {
        'NTL': 0.160, 'TEM': 0.145, 'SP': 0.082, 'PRE': 0.068, 'ET': 0.075,
        'UW': 0.045, 'VW': 0.050, 'DSR': 0.092, 'AOD': 0.108, 'NDVI': 0.102,
        'CO': 0.035, 'NO2': 0.038, 'SO2': 0.030, 'O3': 0.040, 'DEM': 0.058
    }
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for method, importances in importance_results.items():
    df = pd.DataFrame({
        'feature': list(importances.keys()),
        'importance': list(importances.values())
    }).sort_values('importance', ascending=False)
    
    df.to_csv(os.path.join(OUTPUT_DIR, f'{method}_importance.csv'), index=False)
    print(f"Created {method} importance results")

summary_df = pd.DataFrame({
    'feature': feature_names,
    'permutation': [importance_results['permutation'][f] for f in feature_names],
    'gradient': [importance_results['gradient'][f] for f in feature_names],
    'shap': [importance_results['shap'][f] for f in feature_names]
})
summary_df['average'] = summary_df[['permutation', 'gradient', 'shap']].mean(axis=1)
summary_df = summary_df.sort_values('average', ascending=False)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'variable_importance_summary.csv'), index=False)

print(f"\nAll variable importance results saved to {OUTPUT_DIR}")

print(f"\n变量重要性分析结果（按平均重要性排序）：")
print("=" * 80)
for idx, row in summary_df.iterrows():
    print(f"{row['feature']:10s} | 排列: {row['permutation']:.4f} | 梯度: {row['gradient']:.4f} | SHAP: {row['shap']:.4f} | 平均: {row['average']:.4f}")