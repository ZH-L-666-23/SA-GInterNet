import torch
import torch.nn as nn
import torch.nn.functional as F

from .structure_aware_spatial import StructureAwareSpatialBranch
from .cross_graph_temporal import CrossGraphTemporalBranch

class STCGSAN(nn.Module):
    def __init__(self, spatial_input_dim, temporal_input_dim, hidden_dim, output_dim, 
                 spatial_heads=4, temporal_layers=3, fusion_method='concat'):
        super(STCGSAN, self).__init__()
        
        self.spatial_branch = StructureAwareSpatialBranch(
            input_dim=spatial_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=spatial_heads
        )
        
        self.temporal_branch = CrossGraphTemporalBranch(
            input_dim=temporal_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=temporal_layers
        )
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.fusion_layer = nn.Linear(hidden_dim * 2, output_dim)
        elif fusion_method == 'attention':
            self.attention_fusion = AttentionFusion(hidden_dim)
            self.fusion_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.fusion_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, spatial_features, adjacency, temporal_sequence, temporal_edges_list):
        spatial_output = self.spatial_branch(spatial_features, adjacency)
        
        temporal_output = self.temporal_branch(temporal_sequence, temporal_edges_list)
        
        if self.fusion_method == 'concat':
            fused = torch.cat([spatial_output, temporal_output], dim=-1)
            output = self.fusion_layer(fused)
        
        elif self.fusion_method == 'attention':
            fused = self.attention_fusion(spatial_output, temporal_output)
            output = self.fusion_layer(fused)
        
        else:
            fused = (spatial_output + temporal_output) / 2
            output = self.fusion_layer(fused)
        
        return output

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, spatial_out, temporal_out):
        Q = self.query_proj(spatial_out)
        K = self.key_proj(temporal_out)
        V = self.value_proj(temporal_out)
        
        scores = torch.sum(Q * K, dim=-1, keepdim=True) / (Q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=1)
        
        fused = attn * V + (1 - attn) * spatial_out
        return fused