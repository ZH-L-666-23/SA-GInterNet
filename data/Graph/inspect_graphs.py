import numpy as np

graph_path = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\Graph\graph_2017-01.npz'
sequence_path = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\Graph\cross_graph_sequences.npz'

print("=== 单个空间图数据结构 ===")
graph_data = np.load(graph_path, allow_pickle=True)
print(f"Keys: {list(graph_data.keys())}")

print(f"\ncoordinates shape: {graph_data['coordinates'].shape}")
print(f"First 3 coordinates:\n{graph_data['coordinates'][:3]}")

print(f"\nfeatures shape: {graph_data['features'].shape}")
print(f"First 2 feature vectors:\n{graph_data['features'][:2]}")

print(f"\nadjacency shape: {graph_data['adjacency'].shape}")
print(f"Adjacency matrix density: {np.sum(graph_data['adjacency']) / (1000*1000):.4f}")
print(f"First 5 rows of adjacency matrix:\n{graph_data['adjacency'][:5, :10]}")

print(f"\nsample_ids shape: {graph_data['sample_ids'].shape}")
print(f"First 5 sample IDs: {graph_data['sample_ids'][:5]}")

print(f"\ndates shape: {graph_data['dates'].shape}")
print(f"First 5 dates: {graph_data['dates'][:5]}")

print(f"\nprovinces shape: {graph_data['provinces'].shape}")
print(f"Province distribution:")
unique, counts = np.unique(graph_data['provinces'], return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

print("\n=== 跨图时序序列数据结构 ===")
sequence_data = np.load(sequence_path, allow_pickle=True)
print(f"Keys: {list(sequence_data.keys())}")

print(f"\nstart_years shape: {sequence_data['start_years'].shape}")
print(f"start_months shape: {sequence_data['start_months'].shape}")
print(f"end_years shape: {sequence_data['end_years'].shape}")
print(f"end_months shape: {sequence_data['end_months'].shape}")

print(f"\nNumber of sequences: {len(sequence_data['start_years'])}")
print(f"First sequence: {sequence_data['start_years'][0]}-{sequence_data['start_months'][0]:02d} to {sequence_data['end_years'][0]}-{sequence_data['end_months'][0]:02d}")
print(f"Last sequence: {sequence_data['start_years'][-1]}-{sequence_data['start_months'][-1]:02d} to {sequence_data['end_years'][-1]}-{sequence_data['end_months'][-1]:02d}")

print(f"\ngraph_indices: {sequence_data['graph_indices'][0]}")
print(f"temporal_edges count: {len(sequence_data['temporal_edges'][0])}")
if len(sequence_data['temporal_edges'][0]) > 0:
    print(f"First 5 temporal edges:\n{sequence_data['temporal_edges'][0][:5]}")