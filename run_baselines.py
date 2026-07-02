import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.experiments import BaselineRunner

GRAPH_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\Graph'
OUTPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\results\baselines'

class GraphDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        start_idx, mid_idx, end_idx = seq['graph_indices']
        
        def get_year_month(graph_idx):
            year = 2017 + graph_idx // 12
            month = (graph_idx % 12) + 1
            return year, month
        
        start_year, start_month = get_year_month(start_idx)
        mid_year, mid_month = get_year_month(mid_idx)
        end_year, end_month = get_year_month(end_idx)
        
        start_graph = np.load(os.path.join(GRAPH_DIR, f'graph_{start_year}-{start_month:02d}.npz'), allow_pickle=True)
        mid_graph = np.load(os.path.join(GRAPH_DIR, f'graph_{mid_year}-{mid_month:02d}.npz'), allow_pickle=True)
        end_graph = np.load(os.path.join(GRAPH_DIR, f'graph_{end_year}-{end_month:02d}.npz'), allow_pickle=True)
        
        spatial_features = torch.tensor(mid_graph['features'], dtype=torch.float32)
        adjacency = torch.tensor(mid_graph['adjacency'], dtype=torch.float32)
        
        temporal_sequence = torch.stack([
            torch.tensor(start_graph['features'], dtype=torch.float32),
            torch.tensor(mid_graph['features'], dtype=torch.float32)
        ], dim=0)
        
        target = torch.tensor(end_graph['features'][:, 0], dtype=torch.float32).unsqueeze(1)
        
        return {
            'spatial_features': spatial_features,
            'adjacency': adjacency,
            'temporal_sequence': temporal_sequence,
            'temporal_edges': seq['temporal_edges'],
            'target': target
        }

def load_cross_graph_sequences():
    sequence_path = os.path.join(GRAPH_DIR, 'cross_graph_sequences.npz')
    data = np.load(sequence_path, allow_pickle=True)
    
    sequences = []
    for i in range(len(data['start_years'])):
        sequences.append({
            'start_year': int(data['start_years'][i]),
            'start_month': int(data['start_months'][i]),
            'end_year': int(data['end_years'][i]),
            'end_month': int(data['end_months'][i]),
            'graph_indices': data['graph_indices'][i],
            'temporal_edges': data['temporal_edges'][i]
        })
    
    return sequences

class GraphDataLoader:
    def __init__(self, sequences, batch_size=1, shuffle=True):
        self.dataset = GraphDataset(sequences)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_dim = 16
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            
            spatial_features = torch.stack([b['spatial_features'] for b in batch])
            adjacency = torch.stack([b['adjacency'] for b in batch])
            temporal_sequence = torch.stack([b['temporal_sequence'] for b in batch])
            target = torch.stack([b['target'] for b in batch])
            
            yield {
                'spatial_features': spatial_features,
                'adjacency': adjacency,
                'temporal_sequence': temporal_sequence,
                'temporal_edges': [b['temporal_edges'] for b in batch],
                'target': target
            }
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def main():
    print("Loading cross-graph sequences...")
    sequences = load_cross_graph_sequences()
    print(f"Loaded {len(sequences)} sequences")
    
    print("\nCreating data loader...")
    data_loader = GraphDataLoader(sequences, batch_size=1, shuffle=True)
    print(f"Feature dimension: {data_loader.feature_dim}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    print("\nRunning baseline experiments...")
    runner = BaselineRunner(data_loader, device=device)
    
    baseline_types = ['rf', 'xgboost', 'cnn', 'lstm', 'cnn_lstm', 'gcn', 'gat', 'stgcn']
    results = runner.run_baselines(baseline_types)
    
    runner.save_results(results, output_dir=OUTPUT_DIR)
    
    print("\nBaseline experiments completed!")
    print("\nResults summary:")
    for model_name, result in results.items():
        print(f"{model_name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")

if __name__ == '__main__':
    main()