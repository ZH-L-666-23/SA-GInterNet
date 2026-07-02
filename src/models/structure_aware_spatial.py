import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureAwareSpatialBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(StructureAwareSpatialBranch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        self.structure_encoder = StructureEncoder(hidden_dim, num_heads)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim),
            GCNLayer(hidden_dim, hidden_dim)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, features, adjacency):
        x = self.node_embedding(features)
        
        struct_emb = self.structure_encoder(x, adjacency)
        x = x + struct_emb
        
        for gcn in self.gcn_layers:
            x = gcn(x, adjacency)
            x = F.relu(x)
        
        output = self.output_layer(x)
        return output

class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(StructureEncoder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, adjacency):
        batch_size = x.size(0) if x.dim() == 3 else 1
        
        Q = self.query_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        adj_mask = (adjacency == 0).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        scores = scores.masked_fill(adj_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.output_proj(output)
        
        return output

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x, adjacency):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adjacency, support) + self.bias
        return output