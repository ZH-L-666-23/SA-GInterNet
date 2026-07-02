import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

files = [
    '2017.xlsx',
    '2018.xlsx',
    '2019.xlsx',
    '2020.xlsx',
    'spring.xlsx',
    'summer.xlsx',
    'autumn.xlsx',
    'winter.xlsx'
]

titles = [
    '(a) 2017',
    '(b) 2018',
    '(c) 2019',
    '(d) 2020',
    '(e) spring',
    '(f) summer',
    '(g) autumn',
    '(h) winter'
]

stats = [
    {'MAPE': '0.41%', 'RMSE': '2.03', 'MAE': '1.61', 'R2': '0.88', 'N': '5790'},
    {'MAPE': '0.35%', 'RMSE': '1.82', 'MAE': '1.42', 'R2': '0.92', 'N': '6870'},
    {'MAPE': '0.40%', 'RMSE': '1.97', 'MAE': '1.56', 'R2': '0.89', 'N': '9370'},
    {'MAPE': '0.47%', 'RMSE': '2.24', 'MAE': '1.79', 'R2': '0.85', 'N': '10280'},
    {'MAPE': '0.37%', 'RMSE': '1.91', 'MAE': '1.49', 'R2': '0.90', 'N': '4380'},
    {'MAPE': '0.39%', 'RMSE': '2.08', 'MAE': '1.63', 'R2': '0.89', 'N': '5210'},
    {'MAPE': '0.40%', 'RMSE': '2.01', 'MAE': '1.58', 'R2': '0.89', 'N': '14870'},
    {'MAPE': '0.44%', 'RMSE': '2.19', 'MAE': '1.74', 'R2': '0.86', 'N': '8330'}
]

fig = plt.figure(figsize=(24, 10))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])

all_x = []
all_y = []
sc = None

for i, (file, title) in enumerate(zip(files, titles)):
    df = pd.read_excel(file)
    
    if 'real' in df.columns and 'Predicted' in df.columns:
        x = df['real'].values
        y = df['Predicted'].values
    elif len(df.columns) >= 2:
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
    else:
        continue
    
    all_x.extend(x)
    all_y.extend(y)
    
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    sc = ax.scatter(x, y, c=z, s=20, cmap='jet', edgecolor='none')
    
    ax.set_title(title)
    
    if i == 0 or i == 4:
        ax.set_ylabel('Estimated CO2(ppm)', fontsize=13)
    
    if row == 1:
        ax.set_xlabel('Satellite CO2(ppm)', fontsize=13)
    
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
    
    ax.legend()
    
    stat = stats[i]
    text_str = f"MAPE={stat['MAPE']}\nRMSE={stat['RMSE']}\nMAE={stat['MAE']}\nR$^2$={stat['R2']}\nN={stat['N']}"
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

cbar_ax = fig.add_subplot(gs[:, 4])
fig.colorbar(sc, cax=cbar_ax, label='point density')

plt.tight_layout()
plt.savefig('FinalNewnew_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("图像已保存为 scatter_plot.png")