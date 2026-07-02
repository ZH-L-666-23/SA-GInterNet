import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist

class SpatialGraphConstructor:
    def __init__(self, processed_dir='data/processed', output_dir='data/processed'):
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_processed_data(self):
        satellite_path = os.path.join(self.processed_dir, 'satellite_xco2_monthly.csv')
        covariate_path = os.path.join(self.processed_dir, 'covariates_monthly.csv')
        
        if os.path.exists(satellite_path):
            satellite_df = pd.read_csv(satellite_path)
        else:
            raise FileNotFoundError(f"Satellite data not found at {satellite_path}")
        
        if os.path.exists(covariate_path):
            covariate_df = pd.read_csv(covariate_path)
        else:
            raise FileNotFoundError(f"Covariate data not found at {covariate_path}")
        
        merged_df = satellite_df.merge(covariate_df, on=['year', 'month', 'latitude', 'longitude'], how='inner')
        return merged_df
    
    def build_spatial_graph(self, df, year, month, method='distance', threshold=500):
        monthly_data = df[(df['year'] == year) & (df['month'] == month)]
        
        if monthly_data.empty:
            return None
        
        coordinates = monthly_data[['latitude', 'longitude']].values
        node_features = monthly_data.drop(['year', 'month', 'latitude', 'longitude'], axis=1).values
        
        n_nodes = len(monthly_data)
        adjacency = np.zeros((n_nodes, n_nodes))
        
        if method == 'distance':
            distances = cdist(coordinates, coordinates, metric='euclidean')
            adjacency = (distances < threshold).astype(float)
        
        elif method == 'knearest':
            for i in range(n_nodes):
                distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
                nearest_indices = np.argsort(distances)[1:threshold+1]
                adjacency[i, nearest_indices] = 1
        
        elif method == 'delaunay':
            from scipy.spatial import Delaunay
            tri = Delaunay(coordinates)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        adjacency[simplex[i], simplex[j]] = 1
        
        np.fill_diagonal(adjacency, 0)
        
        return {
            'year': year,
            'month': month,
            'nodes': coordinates,
            'features': node_features,
            'adjacency': adjacency,
            'node_ids': monthly_data.index.values
        }
    
    def build_dynamic_spatial_graphs(self, df, start_year=2017, end_year=2020):
        graphs = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                graph = self.build_spatial_graph(df, year, month)
                if graph is not None:
                    graphs.append(graph)
        
        return graphs
    
    def save_graphs(self, graphs, filename='spatial_graphs.npz'):
        output_path = os.path.join(self.output_dir, filename)
        
        graph_data = {
            'years': np.array([g['year'] for g in graphs]),
            'months': np.array([g['month'] for g in graphs]),
            'nodes': [g['nodes'] for g in graphs],
            'features': [g['features'] for g in graphs],
            'adjacency': [g['adjacency'] for g in graphs],
            'node_ids': [g['node_ids'] for g in graphs]
        }
        
        np.savez(output_path, **graph_data)
        print(f"Spatial graphs saved to {output_path}")
        return output_path
    
    def load_graphs(self, filename='spatial_graphs.npz'):
        filepath = os.path.join(self.output_dir, filename)
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

if __name__ == '__main__':
    constructor = SpatialGraphConstructor()
    df = constructor.load_processed_data()
    graphs = constructor.build_dynamic_spatial_graphs(df)
    constructor.save_graphs(graphs)