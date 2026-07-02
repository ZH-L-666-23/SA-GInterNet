import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class TemporalCrossGraphConstructor:
    def __init__(self, processed_dir='data/processed', output_dir='data/processed'):
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_spatial_graphs(self, filename='spatial_graphs.npz'):
        filepath = os.path.join(self.processed_dir, filename)
        data = np.load(filepath, allow_pickle=True)
        
        graphs = []
        for i in range(len(data['years'])):
            graphs.append({
                'year': int(data['years'][i]),
                'month': int(data['months'][i]),
                'nodes': data['nodes'][i],
                'features': data['features'][i],
                'adjacency': data['adjacency'][i],
                'node_ids': data['node_ids'][i]
            })
        
        return graphs
    
    def build_temporal_edges(self, graph_t, graph_t_plus_1, similarity_threshold=0.8):
        nodes_t = graph_t['nodes']
        nodes_t_plus_1 = graph_t_plus_1['nodes']
        
        distances = cdist(nodes_t, nodes_t_plus_1, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        temporal_edges = []
        for i in range(len(nodes_t)):
            nearest_idx = np.argmin(distances[i])
            if min_distances[i] < 1.0:
                temporal_edges.append((i, nearest_idx))
        
        return np.array(temporal_edges)
    
    def build_cross_graph_sequence(self, graphs, window_size=3):
        cross_graph_sequences = []
        
        for i in range(len(graphs) - window_size + 1):
            sequence = graphs[i:i+window_size]
            
            temporal_edges_list = []
            for j in range(window_size - 1):
                edges = self.build_temporal_edges(sequence[j], sequence[j+1])
                temporal_edges_list.append(edges)
            
            cross_graph_sequences.append({
                'start_idx': i,
                'start_year': sequence[0]['year'],
                'start_month': sequence[0]['month'],
                'end_year': sequence[-1]['year'],
                'end_month': sequence[-1]['month'],
                'graphs': sequence,
                'temporal_edges': temporal_edges_list,
                'window_size': window_size
            })
        
        return cross_graph_sequences
    
    def create_input_output_pairs(self, cross_graph_sequences, target_feature='xco2_mean'):
        input_pairs = []
        output_pairs = []
        
        for seq in cross_graph_sequences:
            input_features = []
            for graph in seq['graphs'][:-1]:
                input_features.append(graph['features'])
            
            target_graph = seq['graphs'][-1]
            target_idx = list(target_graph['features'][0]).index(target_feature) if isinstance(target_graph['features'][0], list) else 0
            target_values = target_graph['features'][:, target_idx]
            
            input_pairs.append(input_features)
            output_pairs.append(target_values)
        
        return input_pairs, output_pairs
    
    def save_cross_graph_sequences(self, sequences, filename='cross_graph_sequences.npz'):
        output_path = os.path.join(self.output_dir, filename)
        
        sequence_data = {
            'start_indices': np.array([seq['start_idx'] for seq in sequences]),
            'start_years': np.array([seq['start_year'] for seq in sequences]),
            'start_months': np.array([seq['start_month'] for seq in sequences]),
            'end_years': np.array([seq['end_year'] for seq in sequences]),
            'end_months': np.array([seq['end_month'] for seq in sequences]),
            'window_sizes': np.array([seq['window_size'] for seq in sequences]),
            'graphs': [seq['graphs'] for seq in sequences],
            'temporal_edges': [seq['temporal_edges'] for seq in sequences]
        }
        
        np.savez(output_path, **sequence_data)
        print(f"Cross-graph sequences saved to {output_path}")
        return output_path
    
    def load_cross_graph_sequences(self, filename='cross_graph_sequences.npz'):
        filepath = os.path.join(self.output_dir, filename)
        data = np.load(filepath, allow_pickle=True)
        
        sequences = []
        for i in range(len(data['start_indices'])):
            sequences.append({
                'start_idx': int(data['start_indices'][i]),
                'start_year': int(data['start_years'][i]),
                'start_month': int(data['start_months'][i]),
                'end_year': int(data['end_years'][i]),
                'end_month': int(data['end_months'][i]),
                'window_size': int(data['window_sizes'][i]),
                'graphs': data['graphs'][i],
                'temporal_edges': data['temporal_edges'][i]
            })
        
        return sequences

if __name__ == '__main__':
    constructor = TemporalCrossGraphConstructor()
    graphs = constructor.load_spatial_graphs()
    sequences = constructor.build_cross_graph_sequence(graphs)
    constructor.save_cross_graph_sequences(sequences)