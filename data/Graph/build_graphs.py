import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

INPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\processed'
OUTPUT_DIR = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\Graph'

FEATURE_COLUMNS = ['XCO2', 'NTL', 'TEM', 'SP', 'PRE', 'ET', 'UW', 'VW', 'DSR', 'AOD', 'NDVI', 'CO', 'NO2', 'SO2', 'O3', 'DEM']
PROVINCES = ['Hunan', 'Hubei', 'Anhui', 'Jiangxi']

def load_monthly_data(year, month):
    filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
    filepath = os.path.join(INPUT_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df
    return None

def build_spatial_graph(df, threshold=1.0):
    coordinates = df[['lon', 'lat']].values
    features = df[FEATURE_COLUMNS].values
    
    n_nodes = len(df)
    adjacency = np.zeros((n_nodes, n_nodes))
    
    distances = cdist(coordinates, coordinates, metric='euclidean')
    adjacency = (distances < threshold).astype(float)
    np.fill_diagonal(adjacency, 0)
    
    return {
        'coordinates': coordinates,
        'features': features,
        'adjacency': adjacency,
        'sample_ids': df['sample_id'].values,
        'dates': df['date'].values,
        'provinces': df['province'].values
    }

def build_temporal_edges(graph_t, graph_t_plus_1):
    nodes_t = graph_t['coordinates']
    nodes_t_plus_1 = graph_t_plus_1['coordinates']
    
    distances = cdist(nodes_t, nodes_t_plus_1, metric='euclidean')
    temporal_edges = []
    
    for i in range(len(nodes_t)):
        nearest_idx = np.argmin(distances[i])
        if distances[i, nearest_idx] < 1.0:
            temporal_edges.append((i, nearest_idx))
    
    return np.array(temporal_edges)

def save_graph(graph, year, month):
    filename = f"graph_{year}-{month:02d}.npz"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    np.savez(filepath,
             coordinates=graph['coordinates'],
             features=graph['features'],
             adjacency=graph['adjacency'],
             sample_ids=graph['sample_ids'],
             dates=graph['dates'],
             provinces=graph['provinces'])
    
    print(f"Saved graph: {filename}")

def save_cross_graph_sequences(sequences):
    filename = "cross_graph_sequences.npz"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    sequence_data = {
        'start_years': np.array([seq['start_year'] for seq in sequences]),
        'start_months': np.array([seq['start_month'] for seq in sequences]),
        'end_years': np.array([seq['end_year'] for seq in sequences]),
        'end_months': np.array([seq['end_month'] for seq in sequences]),
        'graph_indices': [seq['graph_indices'] for seq in sequences],
        'temporal_edges': [seq['temporal_edges'] for seq in sequences]
    }
    
    np.savez(filepath, **sequence_data)
    print(f"Saved cross-graph sequences: {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    graphs = []
    all_years = range(2017, 2021)
    all_months = range(1, 13)
    
    for year in all_years:
        for month in all_months:
            df = load_monthly_data(year, month)
            if df is not None:
                graph = build_spatial_graph(df)
                graph['year'] = year
                graph['month'] = month
                graphs.append(graph)
                save_graph(graph, year, month)
    
    print(f"\nBuilt {len(graphs)} spatial graphs")
    
    window_size = 3
    cross_graph_sequences = []
    
    for i in range(len(graphs) - window_size + 1):
        sequence_graphs = graphs[i:i+window_size]
        
        temporal_edges_list = []
        for j in range(window_size - 1):
            edges = build_temporal_edges(sequence_graphs[j], sequence_graphs[j+1])
            temporal_edges_list.append(edges)
        
        cross_graph_sequences.append({
            'start_year': sequence_graphs[0]['year'],
            'start_month': sequence_graphs[0]['month'],
            'end_year': sequence_graphs[-1]['year'],
            'end_month': sequence_graphs[-1]['month'],
            'graph_indices': [i, i+1, i+2],
            'temporal_edges': temporal_edges_list
        })
    
    save_cross_graph_sequences(cross_graph_sequences)
    print(f"Built {len(cross_graph_sequences)} cross-graph sequences")
    
    print_graph_statistics(graphs)

def print_graph_statistics(graphs):
    print("\n=== Graph Statistics ===")
    print(f"Total graphs: {len(graphs)}")
    
    first_graph = graphs[0]
    print(f"\nGraph structure (first graph - 2017-01):")
    print(f"  Number of nodes: {first_graph['coordinates'].shape[0]}")
    print(f"  Feature dimension: {first_graph['features'].shape[1]}")
    print(f"  Adjacency matrix shape: {first_graph['adjacency'].shape}")
    print(f"  Number of edges: {int(np.sum(first_graph['adjacency']))}")
    
    avg_degree = np.mean(np.sum(first_graph['adjacency'], axis=1))
    print(f"  Average node degree: {avg_degree:.2f}")
    
    print(f"\nFeature columns: {FEATURE_COLUMNS}")

if __name__ == '__main__':
    main()