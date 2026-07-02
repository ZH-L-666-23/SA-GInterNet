import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_preprocessing import SatelliteDataProcessor, CovariateDataProcessor
from src.graph_construction import SpatialGraphConstructor, TemporalCrossGraphConstructor
from src.models import STCGSAN
from src.baselines import GCN, GRU, LSTM, GAT, TemporalGNN, MLBaseline
from src.experiments import AblationRunner, BaselineRunner
from src.experiments.variable_importance import VariableImportanceAnalyzer
from src.utils import DataLoaderBuilder, plot_co2_prediction, plot_spatial_map, save_results_to_csv
from config import Config

def preprocess_data():
    print("Preprocessing satellite data...")
    satellite_processor = SatelliteDataProcessor()
    satellite_processor.process_all_satellite_data()
    
    print("Preprocessing covariate data...")
    covariate_processor = CovariateDataProcessor()
    covariate_processor.process_all_covariates()

def construct_graphs():
    print("Constructing spatial graphs...")
    spatial_constructor = SpatialGraphConstructor()
    df = spatial_constructor.load_processed_data()
    graphs = spatial_constructor.build_dynamic_spatial_graphs(df)
    spatial_constructor.save_graphs(graphs)
    
    print("Constructing temporal cross-graph sequences...")
    temporal_constructor = TemporalCrossGraphConstructor()
    sequences = temporal_constructor.build_cross_graph_sequence(graphs)
    temporal_constructor.save_cross_graph_sequences(sequences)
    
    return sequences

def train_st_cgsan(data_loader):
    print("Training ST-CGSAN model...")
    
    model = STCGSAN(
        spatial_input_dim=Config.MODEL['spatial_input_dim'],
        temporal_input_dim=Config.MODEL['temporal_input_dim'],
        hidden_dim=Config.MODEL['hidden_dim'],
        output_dim=Config.MODEL['output_dim'],
        spatial_heads=Config.MODEL['spatial_heads'],
        temporal_layers=Config.MODEL['temporal_layers'],
        fusion_method=Config.MODEL['fusion_method']
    ).to(Config.TRAINING['device'])
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.TRAINING['learning_rate'])
    
    model.train()
    for epoch in range(Config.TRAINING['num_epochs']):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            
            spatial_features = batch['spatial_features'].to(Config.TRAINING['device'])
            adjacency = batch['adjacency'].to(Config.TRAINING['device'])
            temporal_sequence = batch['temporal_sequence'].to(Config.TRAINING['device'])
            temporal_edges = batch['temporal_edges']
            target = batch['target'].to(Config.TRAINING['device'])
            
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
            print(f"Epoch: {epoch+1}/{Config.TRAINING['num_epochs']}, Loss: {total_loss/len(data_loader):.4f}")
    
    return model

def evaluate_model(model, data_loader, cross_graph_sequences):
    print("Evaluating model...")
    
    model.eval()
    predictions = []
    targets = []
    features = []
    nodes = []
    years = []
    months = []
    
    with torch.no_grad():
        for batch in data_loader:
            spatial_features = batch['spatial_features'].to(Config.TRAINING['device'])
            adjacency = batch['adjacency'].to(Config.TRAINING['device'])
            temporal_sequence = batch['temporal_sequence'].to(Config.TRAINING['device'])
            temporal_edges = batch['temporal_edges']
            target = batch['target'].to(Config.TRAINING['device'])
            
            pred = model(
                spatial_features=spatial_features,
                adjacency=adjacency,
                temporal_sequence=temporal_sequence,
                temporal_edges_list=temporal_edges
            )
            
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            features.append(spatial_features.cpu().numpy())
            years.extend(batch['year'])
            months.extend(batch['month'])
    
    for seq in cross_graph_sequences:
        nodes.append(seq['graphs'][-1]['nodes'])
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    features = np.concatenate(features, axis=0)
    
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    results = {
        'predictions': predictions,
        'targets': targets,
        'features': features,
        'rmse': rmse,
        'mae': mae
    }
    
    return results, years, months, nodes

def run_baseline_experiments(data_loader):
    print("Running baseline experiments...")
    runner = BaselineRunner(data_loader, device=Config.TRAINING['device'])
    results = runner.run_baselines(Config.EXPERIMENTS['baseline_types'])
    runner.save_results(results, output_dir=Config.RESULTS['baselines_dir'])
    return results

def run_ablation_experiments(data_loader):
    print("Running ablation experiments...")
    runner = AblationRunner(data_loader, device=Config.TRAINING['device'])
    results = runner.run_ablation(Config.EXPERIMENTS['ablation_types'])
    runner.save_results(results, output_dir=Config.RESULTS['ablation_dir'])
    return results

def run_variable_importance_analysis(model, data_loader):
    print("Running variable importance analysis...")
    analyzer = VariableImportanceAnalyzer(model, Config.FEATURE_NAMES)
    
    for method in Config.EXPERIMENTS['variable_importance_methods']:
        if method == 'permutation':
            importances = analyzer.compute_permutation_importance(data_loader, device=Config.TRAINING['device'])
        elif method == 'gradients':
            importances = analyzer.compute_gradients_importance(data_loader, device=Config.TRAINING['device'])
        elif method == 'shap':
            importances = analyzer.compute_shap_importance(data_loader, device=Config.TRAINING['device'])
        
        analyzer.save_results(importances, method, output_dir=Config.RESULTS['variable_importance_dir'])
    
    return importances

def main():
    preprocess_data()
    cross_graph_sequences = construct_graphs()
    
    data_loader = DataLoaderBuilder(
        cross_graph_sequences,
        feature_dim=Config.MODEL['spatial_input_dim'],
        batch_size=Config.TRAINING['batch_size'],
        shuffle=True
    ).build()
    
    model = train_st_cgsan(data_loader)
    
    results, years, months, nodes = evaluate_model(model, data_loader, cross_graph_sequences)
    
    save_results_to_csv(results, years, months, nodes, output_dir=Config.RESULTS['main_dir'], filename='st_cgsan_predictions.csv')
    
    for i, (year, month) in enumerate(zip(years, months)):
        plot_co2_prediction(results['predictions'][i], results['targets'][i], year, month, output_dir=Config.RESULTS['main_dir'])
        plot_spatial_map(nodes[i][:, 0], nodes[i][:, 1], results['predictions'][i], year, month, output_dir=Config.RESULTS['main_dir'], title='Predicted CO2 Concentration')
        plot_spatial_map(nodes[i][:, 0], nodes[i][:, 1], results['targets'][i], year, month, output_dir=Config.RESULTS['main_dir'], title='True CO2 Concentration')
    
    run_baseline_experiments(data_loader)
    run_ablation_experiments(data_loader)
    run_variable_importance_analysis(model, data_loader)
    
    print("All experiments completed successfully!")

if __name__ == '__main__':
    main()