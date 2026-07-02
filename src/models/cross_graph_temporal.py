import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossGraphTemporalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(CrossGraphTemporalBranch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim, num_layers)
        self.cross_graph_attention = CrossGraphAttention(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, sequence_features, temporal_edges_list):
        encoded_sequence = self.temporal_encoder(sequence_features)
        
        attended = self.cross_graph_attention(encoded_sequence, temporal_edges_list)
        
        output = self.output_layer(attended)
        return output

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, sequence_features):
        if isinstance(sequence_features, list):
            sequence_features = torch.stack(sequence_features, dim=1)
        
        output, _ = self.gru(sequence_features)
        output = self.layer_norm(output)
        
        return output

class CrossGraphAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossGraphAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, encoded_sequence, temporal_edges_list):
        batch_size = encoded_sequence.size(0)
        num_time_steps = encoded_sequence.size(1)
        num_nodes = encoded_sequence.size(2)
        
        Q = self.query_proj(encoded_sequence[:, -1, :, :])
        K = self.key_proj(encoded_sequence[:, :-1, :, :])
        V = self.value_proj(encoded_sequence[:, :-1, :, :])
        
        attention_scores = []
        for t in range(num_time_steps - 1):
            edges = temporal_edges_list[t]
            if edges is not None and len(edges) > 0:
                src_nodes = edges[:, 0]
                tgt_nodes = edges[:, 1]
                
                q = Q[:, tgt_nodes, :]
                k = K[:, t, src_nodes, :]
                
                scores = torch.sum(q * k, dim=-1) / (self.hidden_dim ** 0.5)
                attention_scores.append(scores)
        
        if attention_scores:
            combined_scores = torch.cat(attention_scores, dim=1)
            attn_weights = F.softmax(combined_scores, dim=1)
            
            v_flat = V.view(batch_size, -1, self.hidden_dim)
            attended = torch.bmm(attn_weights.unsqueeze(1), v_flat).squeeze(1)
        else:
            attended = Q.mean(dim=1)
        
        output = self.output_proj(attended)
        return output