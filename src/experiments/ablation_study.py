import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from src.models import STCGSAN, StructureAwareSpatialBranch, CrossGraphTemporalBranch

class AblationModel(nn.Module):
    def __init__(self, spatial_input_dim, temporal_input_dim, hidden_dim, output_dim, ablation_type):
        super(AblationModel, self).__init__()
        self.ablation_type = ablation_type
        
        if ablation_type == 'no_spatial':
            self.temporal_branch = CrossGraphTemporalBranch(
                input_dim=temporal_input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
        
        elif ablation_type == 'no_temporal':
            self.spatial_branch = StructureAwareSpatialBranch(
                input_dim=spatial_input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
        
        elif ablation_type == 'no_structure':
            self.spatial_branch = SimpleGCN(spatial_input_dim, hidden_dim, output_dim)
            self.temporal_branch = CrossGraphTemporalBranch(
                input_dim=temporal_input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            )
            self.fusion_layer = nn.Linear(hidden_dim * 2, output_dim)
        
        elif ablation_type == 'no_cross_graph':
            self.spatial_branch = StructureAwareSpatialBranch(
                input_dim=spatial_input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            )
            self.temporal_branch = SimpleGRU(temporal_input_dim, hidden_dim, hidden_dim)
            self.fusion_layer = nn.Linear(hidden_dim * 2, output_dim)
        
        else:
            self.full_model = STCGSAN(
                spatial_input_dim=spatial_input_dim,
                temporal_input_dim=temporal_input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
    
    def forward(self, spatial_features=None, adjacency=None, temporal_sequence=None, temporal_edges_list=None):
        if self.ablation_type == 'no_spatial':
            return self.temporal_branch(temporal_sequence, temporal_edges_list)
        
        elif self.ablation_type == 'no_temporal':
            return self.spatial_branch(spatial_features, adjacency)
        
        elif self.ablation_type == 'no_structure':
            spatial_out = self.spatial_branch(spatial_features, adjacency)
            temporal_out = self.temporal_branch(temporal_sequence, temporal_edges_list)
            fused = torch.cat([spatial_out, temporal_out], dim=-1)
            return self.fusion_layer(fused)
        
        elif self.ablation_type == 'no_cross_graph':
            spatial_out = self.spatial_branch(spatial_features, adjacency)
            temporal_out = self.temporal_branch(temporal_sequence)
            fused = torch.cat([spatial_out, temporal_out], dim=-1)
            return self.fusion_layer(fused)
        
        else:
            return self.full_model(spatial_features, adjacency, temporal_sequence, temporal_edges_list)

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        x = torch.matmul(adj, self.gcn1(x))
        x = torch.relu(x)
        x = torch.matmul(adj, self.gcn2(x))
        return x

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        _, h_n = self.gru(x)
        return self.output_layer(h_n[-1])

class AblationRunner:
    def __init__(self, data_loader, device='cuda'):
        self.data_loader = data_loader
        self.device = device
        
    def run_ablation(self, ablation_types=['full', 'no_spatial', 'no_temporal', 'no_structure', 'no_cross_graph']):
        results = {}
        
        for ablation_type in ablation_types:
            model = AblationModel(
                spatial_input_dim=self.data_loader.feature_dim,
                temporal_input_dim=self.data_loader.feature_dim,
                hidden_dim=128,
                output_dim=1,
                ablation_type=ablation_type
            ).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            model.train()
            for epoch in range(100):
                total_loss = 0
                for batch in self.data_loader:
                    optimizer.zero_grad()
                    
                    spatial_features = batch['spatial_features'].to(self.device)
                    adjacency = batch['adjacency'].to(self.device)
                    temporal_sequence = batch['temporal_sequence'].to(self.device)
                    temporal_edges = batch['temporal_edges']
                    target = batch['target'].to(self.device)
                    
                    pred = model(
                        spatial_features=spatial_features,
                        adjacency=adjacency,
                        temporal_sequence=temporal_sequence,
                        temporal_edges_list=temporal_edges
                    )
                    
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    print(f"Ablation: {ablation_type}, Epoch: {epoch+1}, Loss: {total_loss/len(self.data_loader):.4f}")
            
            model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in self.data_loader:
                    spatial_features = batch['spatial_features'].to(self.device)
                    adjacency = batch['adjacency'].to(self.device)
                    temporal_sequence = batch['temporal_sequence'].to(self.device)
                    temporal_edges = batch['temporal_edges']
                    target = batch['target'].to(self.device)
                    
                    pred = model(
                        spatial_features=spatial_features,
                        adjacency=adjacency,
                        temporal_sequence=temporal_sequence,
                        temporal_edges_list=temporal_edges
                    )
                    
                    predictions.append(pred.cpu().numpy())
                    targets.append(target.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)
            
            results[ablation_type] = {
                'predictions': predictions,
                'targets': targets,
                'rmse': np.sqrt(np.mean((predictions - targets)**2)),
                'mae': np.mean(np.abs(predictions - targets))
            }
            
            print(f"Ablation {ablation_type} completed - RMSE: {results[ablation_type]['rmse']:.4f}, MAE: {results[ablation_type]['mae']:.4f}")
        
        return results
    
    def save_results(self, results, output_dir='results/ablation'):
        os.makedirs(output_dir, exist_ok=True)
        
        for ablation_type, result in results.items():
            df = pd.DataFrame({
                'predicted': result['predictions'].flatten(),
                'target': result['targets'].flatten()
            })
            df.to_csv(os.path.join(output_dir, f'{ablation_type}_results.csv'), index=False)
        
        summary_df = pd.DataFrame({
            'ablation_type': list(results.keys()),
            'rmse': [r['rmse'] for r in results.values()],
            'mae': [r['mae'] for r in results.values()]
        })
        summary_df.to_csv(os.path.join(output_dir, 'ablation_summary.csv'), index=False)
        
        print(f"Ablation results saved to {output_dir}")