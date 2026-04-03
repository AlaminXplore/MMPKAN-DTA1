import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from hyperparameter import HyperParameter
from mamba_ssm import Mamba

hp = HyperParameter()
os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class MambaBlock(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(
            d_model=embed_size,
            d_state=32,
            d_conv=8,
            expand=4,
        )
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual

class MambaEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        num_layers,
        device,
        dropout,
        max_length,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        
        # Projection layer to match embed_size if needed
        self.input_projection = nn.Linear(input_dim, embed_size) if input_dim != embed_size else nn.Identity()
        
        # Optional positional encoding
        self.use_positional = max_length > 0
        if self.use_positional:
            self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            MambaBlock(embed_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, 128)
    
    def forward(self, x):
        N, seq_length, input_dim = x.shape
        
        # Project input to embed_size if needed
        out = self.input_projection(x)
        
        # Add positional encoding if enabled
        if self.use_positional and seq_length <= self.position_embedding.num_embeddings:
            positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
            pos_embed = self.position_embedding(positions)
            out = out + pos_embed
        
        out = self.dropout(out)
        
        # Apply Mamba layers
        for layer in self.layers:
            out = layer(out)
            
        out = self.fc_out(out)
        return out

class MambaDrug(nn.Module):
    def __init__(
        self,
        input_dim=322,  # Drug embedding dimension
        embed_size=128,
        num_layers=1,
        dropout=0.1,
        device=device,
        max_length=222,  # Max drug sequence length
    ):
        super().__init__()
    
        self.encoder = MambaEncoder(
            input_dim,
            embed_size,
            num_layers,
            device,
            dropout,
            max_length
        )

        self.device = device

    def forward(self, src):
        v = self.encoder(src)
        v= torch.mean(v, 1)
        return v

class MambaProtein(nn.Module):
    def __init__(
        self,
        input_dim=1152,  # Protein embedding dimension
        embed_size=128,
        num_layers=1,
        dropout=0.1,
        device=device,
        max_length=1024,  # Max protein sequence length
    ):
        super().__init__()
    
        self.encoder = MambaEncoder(
            input_dim,
            embed_size,
            num_layers,
            device,
            dropout,
            max_length
        )

        self.device = device

    def forward(self, src):
        v = self.encoder(src)
        v= torch.mean(v, 1)
        return v