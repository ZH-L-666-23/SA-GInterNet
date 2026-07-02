import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from src.baselines import GCN, GRU, LSTM, GAT, CNN, CNN_LSTM, STGCN, TemporalGNN, MLBaseline

class BaselineRunner:
    def __init__(self, data_loader, device='cuda'):
        self.data_loader = data_loader
        self.device = device
    
    def run_baselines(self, baseline_types=['rf', 'xgboost', 'cnn', 'lstm', 'cnn_lstm', 'gcn', 'gat', 'stgcn']):
        results = {}
        
        for baseline_type in baseline_types:
            print(f"Running baseline: {baseline_type}")
            
            if baseline_type in ['gcn', 'gru', 'lstm', 'gat', 'cnn', 'cnn_lstm', 'stgcn', 'temporal_gnn']:
                results[baseline_type] = self.run_nn_baseline(baseline_type)
            else:
                results[baseline_type] = self.run_ml_baseline(baseline_type)
        
        return results
    
    def run_nn_baseline(self, baseline_type):
        input_dim = self.data_loader.feature_dim
        hidden_dim = 128
        output_dim = 1
        
        if baseline_type == 'gcn':
            model = GCN(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'gru':
            model = GRU(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'lstm':
            model = LSTM(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'gat':
            model = GAT(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'cnn':
            model = CNN(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'cnn_lstm':
            model = CNN_LSTM(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'stgcn':
            model = STGCN(input_dim, hidden_dim, output_dim).to(self.device)
        elif baseline_type == 'temporal_gnn':
            model = TemporalGNN(input_dim, hidden_dim, output_dim).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch in self.data_loader:
                optimizer.zero_grad()
                
                if baseline_type == 'gcn' or baseline_type == 'gat':
                    x = batch['spatial_features'].to(self.device)
                    adj = batch['adjacency'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x, adj)
                elif baseline_type == 'gru' or baseline_type == 'lstm':
                    x = batch['temporal_sequence'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x)
                elif baseline_type == 'cnn':
                    x = batch['spatial_features'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x)
                elif baseline_type == 'cnn_lstm':
                    x = batch['temporal_sequence'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x)
                elif baseline_type == 'stgcn':
                    x_list = [batch['temporal_sequence'][:, t, :, :].to(self.device) for t in range(batch['temporal_sequence'].size(1))]
                    adj_list = [batch['adjacency'].to(self.device)] * len(x_list)
                    target = batch['target'].to(self.device)
                    pred = model(x_list, adj_list)
                elif baseline_type == 'temporal_gnn':
                    x_list = [batch['temporal_sequence'][:, t, :, :].to(self.device) for t in range(batch['temporal_sequence'].size(1))]
                    adj_list = [batch['adjacency'].to(self.device)] * len(x_list)
                    target = batch['target'].to(self.device)
                    pred = model(x_list, adj_list)
                
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Baseline: {baseline_type}, Epoch: {epoch+1}, Loss: {total_loss/len(self.data_loader):.4f}")
        
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.data_loader:
                if baseline_type == 'gcn' or baseline_type == 'gat':
                    x = batch['spatial_features'].to(self.device)
                    adj = batch['adjacency'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x, adj)
                elif baseline_type == 'gru' or baseline_type == 'lstm':
                    x = batch['temporal_sequence'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x)
                elif baseline_type == 'cnn':
                    x = batch['spatial_features'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x)
                elif baseline_type == 'cnn_lstm':
                    x = batch['temporal_sequence'].to(self.device)
                    target = batch['target'].to(self.device)
                    pred = model(x)
                elif baseline_type == 'stgcn':
                    x_list = [batch['temporal_sequence'][:, t, :, :].to(self.device) for t in range(batch['temporal_sequence'].size(1))]
                    adj_list = [batch['adjacency'].to(self.device)] * len(x_list)
                    target = batch['target'].to(self.device)
                    pred = model(x_list, adj_list)
                elif baseline_type == 'temporal_gnn':
                    x_list = [batch['temporal_sequence'][:, t, :, :].to(self.device) for t in range(batch['temporal_sequence'].size(1))]
                    adj_list = [batch['adjacency'].to(self.device)] * len(x_list)
                    target = batch['target'].to(self.device)
                    pred = model(x_list, adj_list)
                
                predictions.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        return {
            'predictions': predictions,
            'targets': targets,
            'rmse': np.sqrt(np.mean((predictions - targets)**2)),
            'mae': np.mean(np.abs(predictions - targets))
        }
    
    def run_ml_baseline(self, baseline_type):
        ml_model = MLBaseline(model_type=baseline_type)
        
        X_train = []
        y_train = []
        
        for batch in self.data_loader:
            features = batch['spatial_features'].cpu().numpy()
            target = batch['target'].cpu().numpy()
            
            for i in range(features.shape[0]):
                X_train.append(features[i].flatten())
                y_train.append(target[i].flatten())
        
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()
        
        ml_model.fit(X_train, y_train)
        
        X_test = []
        y_test = []
        
        for batch in self.data_loader:
            features = batch['spatial_features'].cpu().numpy()
            target = batch['target'].cpu().numpy()
            
            for i in range(features.shape[0]):
                X_test.append(features[i].flatten())
                y_test.append(target[i].flatten())
        
        X_test = np.array(X_test)
        y_test = np.array(y_test).flatten()
        
        predictions = ml_model.predict(X_test)
        
        return {
            'predictions': predictions,
            'targets': y_test,
            'rmse': np.sqrt(np.mean((predictions - y_test)**2)),
            'mae': np.mean(np.abs(predictions - y_test))
        }
    
    def save_results(self, results, output_dir='results/baselines'):
        os.makedirs(output_dir, exist_ok=True)
        
        for baseline_type, result in results.items():
            model_dir = os.path.join(output_dir, baseline_type)
            os.makedirs(model_dir, exist_ok=True)
            
            df = pd.DataFrame({
                'predicted': result['predictions'].flatten(),
                'target': result['targets'].flatten()
            })
            df.to_csv(os.path.join(model_dir, f'{baseline_type}_predictions.csv'), index=False)
            
            metrics_df = pd.DataFrame({
                'metric': ['rmse', 'mae'],
                'value': [result['rmse'], result['mae']]
            })
            metrics_df.to_csv(os.path.join(model_dir, f'{baseline_type}_metrics.csv'), index=False)
        
        summary_df = pd.DataFrame({
            'baseline_type': list(results.keys()),
            'rmse': [r['rmse'] for r in results.values()],
            'mae': [r['mae'] for r in results.values()]
        })
        summary_df.to_csv(os.path.join(output_dir, 'baseline_summary.csv'), index=False)
        
        print(f"Baseline results saved to {output_dir}")