import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import HyperParameter
from collections import OrderedDict
import math
from kan import KAN
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
from mamba_encoder import MambaDrug, MambaProtein

hp = HyperParameter()
os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionAttention(nn.Module):

    def __init__(self, in_size, hidden_size=64):
       
        super(FusionAttention, self).__init__()
        self.project_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xt = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_s = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.project_st = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x, xt, s, st):
        
        xtt = self.project_x(torch.cat((xt, s), dim=1))
        xttt = self.project_xt(torch.cat((x, st), dim=1))
        stt = self.project_s(torch.cat((xt, st), dim=1))
        sttt = self.project_st(torch.cat((x, s), dim=1))
        
        a = torch.cat((xtt, xttt, stt, sttt), 1)

        a = torch.softmax(a, dim=1)
        return a


class DrugGraphNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=88,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(DrugGraphNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv0 = GCNConv(num_features_xd, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        self.conv1 = GATConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GATConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GATConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GATConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = GATConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)

        x = self.relu(self.conv0(x, edge_index, edge_weight.mean(dim=1)))
        x = self.bn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class ProteinGraphNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=33,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(ProteinGraphNet, self).__init__()

        dim = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv0 = GCNConv(num_features_xd, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        self.conv1 = GATConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GATConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GATConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GATConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        
        self.conv5 = GATConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)

        x = self.relu(self.conv0(x, edge_index, edge_weight))
        x = self.bn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class MODEL(nn.Module):

    def __init__(self, hp, device):
        super(MODEL, self).__init__()

        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim
        self.drug_max = hp.drug_max_len
        self.prot_max = hp.prot_max_len

        self.dropout = nn.Dropout(0.3)

        self.drug_graph_model = DrugGraphNet(n_output=128)
        self.protein_graph_model = ProteinGraphNet(n_output=128)

        self.drug_encoder = MambaDrug(input_dim=128, max_length=self.drug_max)
        self.prot_encoder = MambaProtein(input_dim=128, max_length=self.prot_max)

        self.drug_ln = nn.LayerNorm(128)
        self.target_ln = nn.LayerNorm(128)

        self.fc2 = nn.Linear(self.protvec_dim, 128)
        self.fc3 = nn.Linear(self.mol2vec_dim, 128)

        self.attnention = FusionAttention(256)
        
        self.kan_block =  KAN([256, 1024, 512, 1])


    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, drug_graph, protein_graph):

        smiles_graph = self.drug_graph_model(drug_graph)    
        fasta_graph = self.protein_graph_model(protein_graph)

        smiles_se = self.drug_encoder(self.fc3(drug_mat))
        smiles_seq = self.drug_ln(smiles_se)

        fasta_se = self.prot_encoder(self.fc2(prot_mat))
        fasta_seq = self.target_ln(fasta_se)

        a = self.attnention(smiles_graph, smiles_seq, fasta_seq, fasta_graph)   
        emb_a  = torch.stack([torch.cat((smiles_seq, fasta_seq), dim=1), torch.cat((smiles_graph, fasta_graph), dim=1), torch.cat((smiles_seq, fasta_graph), dim=1), torch.cat((smiles_graph, fasta_seq), dim=1)], dim=1)
        a = a.unsqueeze(dim=2)
        out = torch.sum(a * emb_a, dim=1)        
        out = self.kan_block(out)

        return out
