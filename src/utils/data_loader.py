import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GraphSequenceDataset(Dataset):
    def __init__(self, cross_graph_sequences, feature_dim):
        self.sequences = cross_graph_sequences
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        spatial_features = torch.tensor(seq['graphs'][-2]['features'], dtype=torch.float32)
        adjacency = torch.tensor(seq['graphs'][-2]['adjacency'], dtype=torch.float32)
        
        temporal_sequence = []
        for i in range(len(seq['graphs']) - 1):
            temporal_sequence.append(torch.tensor(seq['graphs'][i]['features'], dtype=torch.float32))
        
        temporal_sequence = torch.stack(temporal_sequence, dim=0)
        
        target = torch.tensor(seq['graphs'][-1]['features'][:, 0], dtype=torch.float32).unsqueeze(1)
        
        temporal_edges_list = []
        for edges in seq['temporal_edges']:
            if edges is not None and len(edges) > 0:
                temporal_edges_list.append(torch.tensor(edges, dtype=torch.long))
            else:
                temporal_edges_list.append(None)
        
        return {
            'spatial_features': spatial_features,
            'adjacency': adjacency,
            'temporal_sequence': temporal_sequence,
            'temporal_edges': temporal_edges_list,
            'target': target,
            'year': seq['end_year'],
            'month': seq['end_month']
        }

class DataLoaderBuilder:
    def __init__(self, cross_graph_sequences, feature_dim, batch_size=32, shuffle=True):
        self.dataset = GraphSequenceDataset(cross_graph_sequences, feature_dim)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_dim = feature_dim
    
    def build(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        spatial_features = torch.stack([b['spatial_features'] for b in batch])
        adjacency = torch.stack([b['adjacency'] for b in batch])
        temporal_sequence = torch.stack([b['temporal_sequence'] for b in batch])
        target = torch.stack([b['target'] for b in batch])
        years = [b['year'] for b in batch]
        months = [b['month'] for b in batch]
        
        temporal_edges_list = []
        for t in range(len(batch[0]['temporal_edges'])):
            edges_list = [b['temporal_edges'][t] for b in batch if b['temporal_edges'][t] is not None]
            if edges_list:
                temporal_edges_list.append(torch.cat(edges_list, dim=0))
            else:
                temporal_edges_list.append(None)
        
        return {
            'spatial_features': spatial_features,
            'adjacency': adjacency,
            'temporal_sequence': temporal_sequence,
            'temporal_edges': temporal_edges_list,
            'target': target,
            'year': years,
            'month': months
        }