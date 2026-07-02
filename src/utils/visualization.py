import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_co2_prediction(predicted, target, year, month, output_dir='results/main'):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(target, predicted, alpha=0.6, color='b')
    plt.plot([min(target), max(target)], [min(target), max(target)], 'r--')
    plt.xlabel('True CO2 Concentration')
    plt.ylabel('Predicted CO2 Concentration')
    plt.title(f'CO2 Concentration Prediction - {year}-{month:02d}')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f'co2_prediction_{year}_{month:02d}.png'))
    plt.close()

def plot_spatial_map(latitude, longitude, values, year, month, output_dir='results/main', title='CO2 Concentration'):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(longitude, latitude, c=values, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(label='CO2 Concentration (ppm)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'{title} - {year}-{month:02d}')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f'spatial_map_{year}_{month:02d}.png'))
    plt.close()

def plot_variable_importance(feature_names, importances, output_dir='results/variable_importance'):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Variable Importance')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'variable_importance.png'))
    plt.close()

def save_results_to_csv(results, years, months, nodes, output_dir='results/main', filename='predictions.csv'):
    os.makedirs(output_dir, exist_ok=True)
    
    all_rows = []
    for i, (year, month) in enumerate(zip(years, months)):
        for j, node in enumerate(nodes[i]):
            all_rows.append({
                'year': year,
                'month': month,
                'latitude': node[0],
                'longitude': node[1],
                'predicted': results['predictions'][i][j],
                'target': results['targets'][i][j],
                **{f'feature_{k}': results['features'][i][j][k] for k in range(results['features'][i].shape[1])}
            })
    
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Results saved to {os.path.join(output_dir, filename)}")
    
    return df