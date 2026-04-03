import os
import torch
import glob
import os.path as osp
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data, Batch
import networkx as nx
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def matrix_pad_drug(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = torch.zeros((max_len, dim), dtype = torch.float32)
        vec_mask = torch.zeros((max_len), dtype = torch.float32)                            
        new_arr[:len] = arr
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = arr[:max_len]
        vec_mask = torch.ones((max_len), dtype = torch.float32)  
        return new_arr, vec_mask

def matrix_pad_prot(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = torch.zeros((max_len, dim), dtype = torch.float32)
        vec_mask = torch.zeros((max_len), dtype = torch.float32)                            
        new_arr[:len] = torch.from_numpy(arr)
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = torch.from_numpy(arr[:max_len])
        vec_mask = torch.ones((max_len), dtype = torch.float32)  
        return new_arr, vec_mask


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# print(res_weight_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target2graph(distance_map, target_sequence):
    target_edge_index = []
    target_edge_distance = []
    protein_features_esm = seq_feature(target_sequence)
    target_size = distance_map.shape[0]

    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    index_row, index_col = np.where(distance_map >= 0.5)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
        target_edge_distance.append(distance_map[i, j])

    target_node_feature = torch.FloatTensor(protein_features_esm)
    target_edge_index = torch.LongTensor(target_edge_index).transpose(1, 0)
    target_edge_distance = torch.FloatTensor(target_edge_distance)

    return target_size, target_node_feature, target_edge_index, target_edge_distance
 


def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                                                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                'Pt', 'Hg', 'Pb', 'X']]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['degree'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['ImplicitValence'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['num_h'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['hybridization'] == x) for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))


    return edge_index, edge_attr

def smile2graph(smile):
    mol = Chem.MolFromSmiles(smile)

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    mol_size = mol.GetNumAtoms()
    g = nx.DiGraph()
    
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                a_type=atom_i.GetSymbol(),
                a_num=atom_i.GetAtomicNum(),
                acceptor=0,
                donor=0,
                aromatic=atom_i.GetIsAromatic(),
                hybridization=atom_i.GetHybridization(),
                num_h=atom_i.GetTotalNumHs(),
                degree = atom_i.GetDegree(),
                # 5 more node features
                ExplicitValence=atom_i.GetExplicitValence(),
                FormalCharge=atom_i.GetFormalCharge(),
                ImplicitValence=atom_i.GetImplicitValence(),
                NumExplicitHs=atom_i.GetNumExplicitHs(),
                NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
            )
            
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor']

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                      
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )
                
    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)         

    return mol_size, node_attr, edge_index, edge_attr

def my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map, isEsm=False):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_vec = torch.zeros((batch_size, mol2vec_dim), dtype=torch.float32)
    b_prot_vec = torch.zeros((batch_size, protvec_dim), dtype=torch.float32)
    b_drug_mask = torch.zeros((batch_size, drug_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    b_label = torch.zeros(batch_size, dtype=torch.float32)
    
    b_drug_graph = []    
    b_protein_graph = []
    
    # Process each sample in the batch
    for i, pair in enumerate(batch_data):        
        drug_id, prot_id, label = pair[0], pair[2], pair[4]
        drug_smiles = drug_df.loc[drug_df['drug_key'] == drug_id, 'compound_iso_smiles'].iloc[0]
        prot_seq = prot_df.loc[prot_df['target_key'] == prot_id, 'target_sequence'].iloc[0]        
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_vec = mol2vec_dict["vec_dict"][drug_id]
        prot_vec = protvec_dict["vec_dict"][prot_id]
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id]
        drug_mat_pad, drug_mask = matrix_pad_drug(drug_mat, drug_max)        
        prot_mat_pad, prot_mask = matrix_pad_prot(prot_mat, protein_max) 

        # Drug graph for PyTorch Geometric
        mol_size, node_attr, edge_index, edge_attr = smile2graph(drug_smiles)
        drug_graph = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)
        b_drug_graph.append(drug_graph)
        
        target_size, target_features, target_edge_index, target_edge_distance = target2graph(prot_contact_map, prot_seq)
        protein_graph = Data(x=target_features, edge_index=target_edge_index, edge_weight=target_edge_distance)
        b_protein_graph.append(protein_graph)
        
        
        # Store other values for the batch
        b_drug_vec[i] = drug_vec
        b_prot_vec[i] = torch.from_numpy(prot_vec)
        b_drug_mat[i] = drug_mat_pad
        b_drug_mask[i] = drug_mask
        b_prot_mat[i] = prot_mat_pad
        b_prot_mask[i] = prot_mask
        b_label[i] = label
    
    
    
    # Batch graphs using PyG's built-in functionality
    b_drug_graph = Batch.from_data_list(b_drug_graph)
    b_protein_graph = Batch.from_data_list(b_protein_graph)
    
    return b_drug_vec, b_prot_vec, b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_drug_graph, b_protein_graph, b_label

class CustomDataSet(Dataset):
    def __init__(self, dataset, hp):    
        self.hp = hp
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset.iloc[index,:]

    def __len__(self):
        return len(self.dataset)
