import torch
import numpy as np
import pandas as pd
import os

class VariableImportanceAnalyzer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def compute_permutation_importance(self, data_loader, device='cuda', n_permutations=5):
        self.model.eval()
        
        original_predictions = []
        with torch.no_grad():
            for batch in data_loader:
                spatial_features = batch['spatial_features'].to(device)
                adjacency = batch['adjacency'].to(device)
                temporal_sequence = batch['temporal_sequence'].to(device)
                temporal_edges = batch['temporal_edges']
                
                pred = self.model(
                    spatial_features=spatial_features,
                    adjacency=adjacency,
                    temporal_sequence=temporal_sequence,
                    temporal_edges_list=temporal_edges
                )
                original_predictions.append(pred.cpu().numpy())
        
        original_predictions = np.concatenate(original_predictions, axis=0)
        targets = []
        for batch in data_loader:
            targets.append(batch['target'].cpu().numpy())
        targets = np.concatenate(targets, axis=0)
        
        original_rmse = np.sqrt(np.mean((original_predictions - targets)**2))
        
        importances = []
        
        for feature_idx in range(len(self.feature_names)):
            permuted_rmses = []
            
            for _ in range(n_permutations):
                permuted_predictions = []
                
                with torch.no_grad():
                    for batch in data_loader:
                        spatial_features = batch['spatial_features'].clone().to(device)
                        adjacency = batch['adjacency'].to(device)
                        temporal_sequence = batch['temporal_sequence'].clone().to(device)
                        temporal_edges = batch['temporal_edges']
                        
                        spatial_features[:, :, feature_idx] = torch.randperm(spatial_features.size(1)).to(device)
                        for t in range(temporal_sequence.size(1)):
                            temporal_sequence[:, t, :, feature_idx] = torch.randperm(temporal_sequence.size(2)).to(device)
                        
                        pred = self.model(
                            spatial_features=spatial_features,
                            adjacency=adjacency,
                            temporal_sequence=temporal_sequence,
                            temporal_edges_list=temporal_edges
                        )
                        permuted_predictions.append(pred.cpu().numpy())
                
                permuted_predictions = np.concatenate(permuted_predictions, axis=0)
                permuted_rmse = np.sqrt(np.mean((permuted_predictions - targets)**2))
                permuted_rmses.append(permuted_rmse)
            
            avg_permuted_rmse = np.mean(permuted_rmses)
            importance = avg_permuted_rmse - original_rmse
            importances.append(importance)
        
        importances = np.array(importances)
        importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
        return importances
    
    def compute_gradients_importance(self, data_loader, device='cuda'):
        self.model.train()
        gradients_sum = np.zeros(len(self.feature_names))
        n_samples = 0
        
        for batch in data_loader:
            self.model.zero_grad()
            
            spatial_features = batch['spatial_features'].to(device).requires_grad_(True)
            adjacency = batch['adjacency'].to(device)
            temporal_sequence = batch['temporal_sequence'].to(device).requires_grad_(True)
            temporal_edges = batch['temporal_edges']
            target = batch['target'].to(device)
            
            pred = self.model(
                spatial_features=spatial_features,
                adjacency=adjacency,
                temporal_sequence=temporal_sequence,
                temporal_edges_list=temporal_edges
            )
            
            loss = torch.mean((pred - target)**2)
            loss.backward()
            
            if spatial_features.grad is not None:
                gradients_sum += np.mean(np.abs(spatial_features.grad.cpu().numpy()), axis=(0, 1))
            if temporal_sequence.grad is not None:
                gradients_sum += np.mean(np.abs(temporal_sequence.grad.cpu().numpy()), axis=(0, 1, 2))
            
            n_samples += 1
        
        importances = gradients_sum / n_samples
        importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
        return importances
    
    def compute_shap_importance(self, data_loader, device='cuda', n_samples=100):
        self.model.eval()
        
        background_data = []
        for batch in data_loader:
            spatial_features = batch['spatial_features'].cpu().numpy()
            temporal_sequence = batch['temporal_sequence'].cpu().numpy()
            background_data.append((spatial_features, temporal_sequence))
        
        if not background_data:
            return np.zeros(len(self.feature_names))
        
        spatial_bg = np.concatenate([d[0] for d in background_data], axis=0)
        temporal_bg = np.concatenate([d[1] for d in background_data], axis=0)
        
        shap_values = np.zeros(len(self.feature_names))
        
        for _ in range(n_samples):
            idx = np.random.randint(0, spatial_bg.shape[0])
            x = spatial_bg[idx]
            x_temporal = temporal_bg[idx]
            
            base_pred = self._predict_with_mask(x, x_temporal, np.zeros(len(self.feature_names)), device)
            full_pred = self._predict_with_mask(x, x_temporal, np.ones(len(self.feature_names)), device)
            
            for i in range(len(self.feature_names)):
                mask = np.ones(len(self.feature_names))
                mask[i] = 0
                masked_pred = self._predict_with_mask(x, x_temporal, mask, device)
                shap_values[i] += (full_pred - masked_pred) / n_samples
        
        shap_values = shap_values / np.sum(np.abs(shap_values)) if np.sum(np.abs(shap_values)) > 0 else shap_values
        
        return shap_values
    
    def _predict_with_mask(self, x, x_temporal, mask, device):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        x_temporal_tensor = torch.tensor(x_temporal, dtype=torch.float32).unsqueeze(0).to(device)
        
        x_tensor[:, :, mask == 0] = 0
        x_temporal_tensor[:, :, :, mask == 0] = 0
        
        adjacency = torch.eye(x.shape[0]).unsqueeze(0).to(device)
        temporal_edges = None
        
        with torch.no_grad():
            pred = self.model(
                spatial_features=x_tensor,
                adjacency=adjacency,
                temporal_sequence=x_temporal_tensor,
                temporal_edges_list=temporal_edges
            )
        
        return pred.cpu().numpy()[0, 0]
    
    def save_results(self, importances, method_name, output_dir='results/variable_importance'):
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        df.to_csv(os.path.join(output_dir, f'{method_name}_importance.csv'), index=False)
        print(f"Variable importance results saved to {output_dir}/{method_name}_importance.csv")
        
        return df