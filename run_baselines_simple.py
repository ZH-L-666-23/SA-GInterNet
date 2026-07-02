import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

INPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\processed'
OUTPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\results\baselines'

FEATURE_COLS = ['NTL', 'TEM', 'SP', 'PRE', 'ET', 'UW', 'VW', 'DSR', 'AOD', 'NDVI', 'CO', 'NO2', 'SO2', 'O3', 'DEM']
TARGET_COL = 'XCO2'

def load_all_data():
    all_data = []
    for year in range(2017, 2021):
        for month in range(1, 13):
            filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
            filepath = os.path.join(INPUT_DIR, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['year'] = year
                df['month'] = month
                all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def run_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'predictions': preds,
        'targets': y_test,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_xgboost(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'predictions': preds,
        'targets': y_test,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_cnn(X_train, y_train, X_test, y_test, epochs=50):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    model = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(192, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_tensor)
                val_loss = criterion(val_preds, y_test_tensor)
            print(f"CNN Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
    
    return {
        'predictions': preds.flatten(),
        'targets': y_test.values,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_lstm(X_train, y_train, X_test, y_test, epochs=50):
    X_train_seq = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_seq = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    model = nn.Sequential(
        nn.LSTM(15, 64, batch_first=True),
        nn.Linear(64, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds, _ = model(X_train_seq)
        preds = preds[:, -1, :]
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds, _ = model(X_test_seq)
                val_preds = val_preds[:, -1, :]
                val_loss = criterion(val_preds, y_test_tensor)
            print(f"LSTM Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        preds, _ = model(X_test_seq)
        preds = preds[:, -1, :].numpy()
    
    return {
        'predictions': preds.flatten(),
        'targets': y_test.values,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_cnn_lstm(X_train, y_train, X_test, y_test, epochs=50):
    X_train_seq = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_seq = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    cnn = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2)
    )
    
    lstm = nn.LSTM(192, 64, batch_first=True)
    fc = nn.Linear(64, 1)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(lstm.parameters()) + list(fc.parameters()), lr=1e-3)
    
    for epoch in range(epochs):
        cnn.train()
        lstm.train()
        optimizer.zero_grad()
        
        cnn_out = cnn(X_train_seq)
        cnn_out = cnn_out.view(cnn_out.size(0), 1, -1)
        lstm_out, _ = lstm(cnn_out)
        preds = fc(lstm_out[:, -1, :])
        
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            cnn.eval()
            lstm.eval()
            with torch.no_grad():
                cnn_out = cnn(X_test_seq)
                cnn_out = cnn_out.view(cnn_out.size(0), 1, -1)
                lstm_out, _ = lstm(cnn_out)
                val_preds = fc(lstm_out[:, -1, :])
                val_loss = criterion(val_preds, y_test_tensor)
            print(f"CNN-LSTM Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    cnn.eval()
    lstm.eval()
    with torch.no_grad():
        cnn_out = cnn(X_test_seq)
        cnn_out = cnn_out.view(cnn_out.size(0), 1, -1)
        lstm_out, _ = lstm(cnn_out)
        preds = fc(lstm_out[:, -1, :]).numpy()
    
    return {
        'predictions': preds.flatten(),
        'targets': y_test.values,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_gcn(X_train, y_train, X_test, y_test, epochs=50):
    adj_train = np.ones((len(X_train), len(X_train))) * 0.1
    np.fill_diagonal(adj_train, 0)
    adj_test = np.ones((len(X_test), len(X_test))) * 0.1
    np.fill_diagonal(adj_test, 0)
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    adj_train_tensor = torch.tensor(adj_train, dtype=torch.float32)
    adj_test_tensor = torch.tensor(adj_test, dtype=torch.float32)
    
    class GCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x, adj):
            x = torch.matmul(adj, x)
            x = torch.relu(self.fc1(x))
            x = torch.matmul(adj, x)
            x = self.fc2(x)
            return x
    
    model = GCN(15, 64, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor, adj_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_tensor, adj_test_tensor)
                val_loss = criterion(val_preds, y_test_tensor)
            print(f"GCN Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor, adj_test_tensor).numpy()
    
    return {
        'predictions': preds.flatten(),
        'targets': y_test.values,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_gat(X_train, y_train, X_test, y_test, epochs=50):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    class GAT(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
            super().__init__()
            self.query = nn.Linear(input_dim, hidden_dim)
            self.key = nn.Linear(input_dim, hidden_dim)
            self.value = nn.Linear(input_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.heads = heads
        
        def forward(self, x):
            q = self.query(x).view(-1, self.heads, -1)
            k = self.key(x).view(-1, self.heads, -1)
            v = self.value(x).view(-1, self.heads, -1)
            
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1)), dim=-1)
            out = torch.matmul(attn, v)
            out = out.view(-1, self.heads * (hidden_dim // self.heads))
            out = self.fc(out)
            return out
    
    model = GAT(15, 64, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_tensor)
                val_loss = criterion(val_preds, y_test_tensor)
            print(f"GAT Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
    
    return {
        'predictions': preds.flatten(),
        'targets': y_test.values,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def run_stgcn(X_train, y_train, X_test, y_test, epochs=50):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    class STGCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = x.transpose(1, 2)
            x = torch.relu(self.conv(x))
            x = x.transpose(1, 2)
            x, _ = self.lstm(x)
            x = self.fc(x[:, -1, :])
            return x
    
    model = STGCN(15, 64, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_tensor)
                val_loss = criterion(val_preds, y_test_tensor)
            print(f"STGCN Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
    
    return {
        'predictions': preds.flatten(),
        'targets': y_test.values,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mae': mean_absolute_error(y_test, preds)
    }

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    summary_rows = []
    for model_name, result in results.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        df = pd.DataFrame({
            'predicted': result['predictions'],
            'target': result['targets']
        })
        df.to_csv(os.path.join(model_dir, f'{model_name}_predictions.csv'), index=False)
        
        metrics_df = pd.DataFrame({
            'metric': ['rmse', 'mae'],
            'value': [result['rmse'], result['mae']]
        })
        metrics_df.to_csv(os.path.join(model_dir, f'{model_name}_metrics.csv'), index=False)
        
        summary_rows.append({
            'model': model_name,
            'rmse': result['rmse'],
            'mae': result['mae']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'baseline_summary.csv'), index=False)
    print(f"\nResults saved to {output_dir}")

def main():
    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} samples")
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    results = {}
    
    print("\nRunning Random Forest...")
    results['rf'] = run_rf(X_train, y_train, X_test, y_test)
    print(f"RF: RMSE={results['rf']['rmse']:.4f}, MAE={results['rf']['mae']:.4f}")
    
    print("\nRunning XGBoost...")
    results['xgboost'] = run_xgboost(X_train, y_train, X_test, y_test)
    print(f"XGBoost: RMSE={results['xgboost']['rmse']:.4f}, MAE={results['xgboost']['mae']:.4f}")
    
    print("\nRunning CNN...")
    results['cnn'] = run_cnn(X_train, y_train, X_test, y_test)
    print(f"CNN: RMSE={results['cnn']['rmse']:.4f}, MAE={results['cnn']['mae']:.4f}")
    
    print("\nRunning LSTM...")
    results['lstm'] = run_lstm(X_train, y_train, X_test, y_test)
    print(f"LSTM: RMSE={results['lstm']['rmse']:.4f}, MAE={results['lstm']['mae']:.4f}")
    
    print("\nRunning CNN-LSTM...")
    results['cnn_lstm'] = run_cnn_lstm(X_train, y_train, X_test, y_test)
    print(f"CNN-LSTM: RMSE={results['cnn_lstm']['rmse']:.4f}, MAE={results['cnn_lstm']['mae']:.4f}")
    
    print("\nRunning GCN...")
    results['gcn'] = run_gcn(X_train, y_train, X_test, y_test)
    print(f"GCN: RMSE={results['gcn']['rmse']:.4f}, MAE={results['gcn']['mae']:.4f}")
    
    print("\nRunning GAT...")
    results['gat'] = run_gat(X_train, y_train, X_test, y_test)
    print(f"GAT: RMSE={results['gat']['rmse']:.4f}, MAE={results['gat']['mae']:.4f}")
    
    print("\nRunning STGCN...")
    results['stgcn'] = run_stgcn(X_train, y_train, X_test, y_test)
    print(f"STGCN: RMSE={results['stgcn']['rmse']:.4f}, MAE={results['stgcn']['mae']:.4f}")
    
    save_results(results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Baseline Experiments Summary:")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name:12s} | RMSE: {result['rmse']:.4f} | MAE: {result['mae']:.4f}")

if __name__ == '__main__':
    main()