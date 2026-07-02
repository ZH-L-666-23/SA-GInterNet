class Config:
    DATA = {
        'raw_satellite_dir': 'data/raw/satellite',
        'raw_covariates_dir': 'data/raw/covariates',
        'processed_dir': 'data/processed',
        'satellite_file': 'satellite_xco2_monthly.csv',
        'covariates_file': 'covariates_monthly.csv',
        'spatial_graphs_file': 'spatial_graphs.npz',
        'cross_graph_sequences_file': 'cross_graph_sequences.npz'
    }
    
    MODEL = {
        'spatial_input_dim': 12,
        'temporal_input_dim': 12,
        'hidden_dim': 128,
        'output_dim': 1,
        'spatial_heads': 4,
        'temporal_layers': 3,
        'fusion_method': 'concat'
    }
    
    TRAINING = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    EXPERIMENTS = {
        'baseline_types': ['gcn', 'gru', 'lstm', 'gat', 'temporal_gnn', 'linear', 'rf', 'svr'],
        'ablation_types': ['full', 'no_spatial', 'no_temporal', 'no_structure', 'no_cross_graph'],
        'variable_importance_methods': ['permutation', 'gradients', 'shap']
    }
    
    RESULTS = {
        'main_dir': 'results/main',
        'baselines_dir': 'results/baselines',
        'ablation_dir': 'results/ablation',
        'variable_importance_dir': 'results/variable_importance'
    }
    
    FEATURE_NAMES = [
        'xco2_mean',
        'xco2_std',
        'xco2_count',
        'xco2_uncertainty_mean',
        'temperature',
        'co2_flux',
        'ndvi',
        'pressure',
        'humidity',
        'wind_speed',
        'solar_radiation',
        'land_use'
    ]

try:
    import torch
except ImportError:
    Config.TRAINING['device'] = 'cpu'