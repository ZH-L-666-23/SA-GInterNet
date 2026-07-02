import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        for layer in self.layers:
            x = torch.matmul(adj, x)
            x = layer(x)
            x = F.relu(x)
        
        return self.output_layer(x)

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        
        _, h_n = self.gru(x)
        return self.output_layer(h_n[-1])

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        
        _, (h_n, _) = self.lstm(x)
        return self.output_layer(h_n[-1])

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        Q = self.query_proj(x).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.key_proj(x).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.value_proj(x).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        adj_mask = (adj == 0).unsqueeze(0).repeat(self.num_heads, 1, 1)
        scores = scores.masked_fill(adj_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        output = output.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)
        return self.output_layer(output)

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        
        batch_size, seq_len, num_nodes, feat_dim = x.shape
        x = x.view(batch_size * seq_len, num_nodes, feat_dim)
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x.view(batch_size, seq_len, num_nodes, -1)
        x = x.mean(dim=2)
        
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class STGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STGCN, self).__init__()
        self.gcn1 = GCN(input_dim, hidden_dim, hidden_dim)
        self.gcn2 = GCN(hidden_dim, hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x_list, adj_list):
        gcn_outputs = []
        for x, adj in zip(x_list, adj_list):
            x = self.gcn1(x, adj)
            x = F.relu(x)
            x = self.gcn2(x, adj)
            x = F.relu(x)
            gcn_outputs.append(x.mean(dim=0))
        
        gcn_outputs = torch.stack(gcn_outputs, dim=0).unsqueeze(0)
        _, h_n = self.lstm(gcn_outputs)
        return self.fc(h_n[-1])

class TemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TemporalGNN, self).__init__()
        self.gcn_layers = nn.ModuleList([GCN(input_dim, hidden_dim, hidden_dim)])
        
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCN(hidden_dim, hidden_dim, hidden_dim))
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x_list, adj_list):
        gcn_outputs = []
        for x, adj in zip(x_list, adj_list):
            for gcn in self.gcn_layers:
                x = gcn(x, adj)
                x = F.relu(x)
            gcn_outputs.append(x)
        
        gcn_outputs = torch.stack(gcn_outputs, dim=1)
        _, h_n = self.gru(gcn_outputs)
        return self.output_layer(h_n[-1])

class MLBaseline:
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100)
        elif model_type == 'svr':
            self.model = SVR()
        elif model_type == 'xgboost':
            self.model = GradientBoostingRegressor(n_estimators=100)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)