import os
import random
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn

import csv
from metrics import calculate_metrics, get_mse

import warnings
warnings.filterwarnings("ignore")

def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)
    
def test(model, dataloader):
    model.eval()
    preds = []
    labels = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data

        mol_vec = mol_vec.to(device)
        prot_vec = prot_vec.to(device)
        mol_mat = mol_mat.to(device)
        mol_mat_mask = mol_mat_mask.to(device)
        prot_mat = prot_mat.to(device)
        prot_mat_mask = prot_mat_mask.to(device)
        drugh_graph = drugh_graph.to(device)
        protein_graph = protein_graph.to(device)
        affinity = affinity.to(device)

        with torch.no_grad():
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse, ci, rm2 = calculate_metrics(labels, preds)
    return mse, ci, rm2, preds, labels  # CHANGED: Now returns 5 values instead of 3

def save_training_log(train_log, hp, fold_i, log_dir):
    with open(log_dir, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mse", "ci", "rm2"])
        for epoch, (mse, ci, rm2) in enumerate(train_log, 1):
            writer.writerow([epoch, mse, ci, rm2])
    print(f'Saved training log at {log_dir}')

def save_comprehensive_test_log(fold_results, hp, log_dir):
    # Extract metrics arrays
    mse_values = [r['mse'] for r in fold_results]
    ci_values = [r['ci'] for r in fold_results]
    rm2_values = [r['rm2'] for r in fold_results]
    
    # Calculate statistics
    mse_mean = np.mean(mse_values)
    mse_var = np.var(mse_values)
    mse_std = np.std(mse_values)
    
    ci_mean = np.mean(ci_values)
    ci_var = np.var(ci_values)
    ci_std = np.std(ci_values)
    
    rm2_mean = np.mean(rm2_values)
    rm2_var = np.var(rm2_values)
    rm2_std = np.std(rm2_values)
    dataset_str = f"{hp.dataset}-{hp.running_set}"

    with open(log_dir, "w", newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["type", dataset_str, "mse", "ci", "rm2"])
        
        # Write fold results
        for result in fold_results:
            writer.writerow([
                f"fold_{result['fold']}",
                dataset_str,
                f"{result['mse']:.4f}",
                f"{result['ci']:.4f}",
                f"{result['rm2']:.4f}"
            ])
        
        writer.writerow(["Summary statistics:"])
        writer.writerow(["Metric", "mean:", "std:", "var:"])
        writer.writerow(["mse:", f"{mse_mean:.4f}", f"{mse_std:.4f}", f"{mse_var:.4f}"])
        writer.writerow(["ci:", f"{ci_mean:.4f}", f"{ci_std:.4f}", f"{ci_var:.4f}"])
        writer.writerow(["rm2:", f"{rm2_mean:.4f}", f"{rm2_std:.4f}", f"{rm2_var:.4f}"])
    
    print(f'Saved comprehensive test log at {log_dir}')
    
    # Return statistics dictionary
    return {
        'mse': {'mean': mse_mean, 'std': mse_std, 'var': mse_var},
        'ci': {'mean': ci_mean, 'std': ci_std, 'var': ci_var},
        'rm2': {'mean': rm2_mean, 'std': rm2_std, 'var': rm2_var}
    }

if __name__ == "__main__":
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Dataset: {hp.dataset}-{hp.running_set}")
    print(f"Pretrain: {hp.mol2vec_dir}-{hp.protvec_dir}")
    
    # Create log directory if it doesn't exist
    log_base_dir = f"/homeb/habibulla/Alamin/KHANkibaW/log"
    os.makedirs(log_base_dir, exist_ok=True)
    
    fold_results = []  # List to store results for each fold
    all_test_predictions = []
    all_test_labels = []
    
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    contact_map = load_pickle(hp.contact_map)
    
    for fold_i in range(1, hp.kfold + 1):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_i}")
        print(f"{'='*60}")
        
        train_dir = os.path.join(dataset_root, f'fold_{fold_i}_train.csv')
        valid_dir = os.path.join(dataset_root, f'fold_{fold_i}_valid.csv')
        test_dir = os.path.join(dataset_root, f'fold_{fold_i}_test.csv')
        
        train_set = CustomDataSet(pd.read_csv(train_dir), hp)
        valid_set = CustomDataSet(pd.read_csv(valid_dir), hp)
        test_set = CustomDataSet(pd.read_csv(test_dir), hp)
        
        train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=12, 
                                       collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
        valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=12, 
                                       collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
        test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=12, 
                                      collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
        print("Dataset loaded successfully")  # CHANGED: Improved message
    
        model = nn.DataParallel(Model(hp, device))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
        criterion = F.mse_loss
    
        train_log = []     
        best_valid_mse = 10  
        patience = 0    
        model_fromTrain = f'./MMPKAN-DTA/savemodel/{hp.dataset}-{hp.running_set}-fold{fold_i}-{hp.current_time}.pth'
        
        for epoch in range(1, hp.Epoch + 1):
            # training
            model.train()
            pred = []
            label = []             
            for batch_data in train_dataset_load:
                mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data

                mol_vec = mol_vec.to(device)
                prot_vec = prot_vec.to(device)
                mol_mat = mol_mat.to(device)
                mol_mat_mask = mol_mat_mask.to(device)
                prot_mat = prot_mat.to(device)
                prot_mat_mask = prot_mat_mask.to(device)
                drugh_graph = drugh_graph.to(device)
                protein_graph = protein_graph.to(device)
                affinity = affinity.to(device)
                  
                predictions = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
                pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
                label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()            
                
                loss = criterion(predictions.squeeze(), affinity)
                loss.backward()                
                optimizer.step()
                optimizer.zero_grad()
            
            pred = np.array(pred)
            label = np.array(label)
            mse_value, ci_value, rm2_value = calculate_metrics(pred, label)
            train_log.append([mse_value, ci_value, rm2_value])
            print(f'Fold {fold_i}, Epoch {epoch}: Train MSE: {mse_value:.4f}, CI: {ci_value:.4f}, RM2: {rm2_value:.4f}')
            
            # valid
            mse, ci, rm2, _, _ = test(model, valid_dataset_load)
            print(f'Fold {fold_i}, Epoch {epoch}: Valid MSE: {mse:.4f}, CI: {ci:.4f}, RM2: {rm2:.4f}')
             
            # Early stop        
            if mse < best_valid_mse:
                patience = 0
                best_valid_mse = mse
                # save model
                torch.save(model.state_dict(), model_fromTrain)
                print(f'New best model saved: Valid MSE: {mse:.4f}')
            else:
                patience += 1
                if patience > hp.max_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                    
        training_log_path = os.path.join(log_base_dir, f"training_log_fold{fold_i}_{hp.dataset}_{hp.running_set}_{hp.current_time}.csv")
        save_training_log(train_log, hp, fold_i, training_log_path)
        
        # Test
        print(f"\nTesting fold {fold_i}...")
        predModel = nn.DataParallel(Model(hp, device))
        predModel.load_state_dict(torch.load(model_fromTrain))
        predModel = predModel.to(device)    
        mse, ci, rm2, preds, labels = test(predModel, test_dataset_load)  # CHANGED: Now unpacks 5 values
        print(f'Fold {fold_i} Test Results: MSE: {mse:.4f}, CI: {ci:.4f}, RM2: {rm2:.4f}\n')
        
        fold_results.append({
            'fold': fold_i,
            'mse': mse,
            'ci': ci,
            'rm2': rm2
        })
        all_test_predictions.extend(preds)
        all_test_labels.extend(labels)
        
    test_log_path = os.path.join(log_base_dir, f"test_log_{hp.dataset}_{hp.running_set}_{hp.current_time}.csv")
    statistics = save_comprehensive_test_log(fold_results, hp, test_log_path)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST RESULTS - {hp.dataset}-{hp.running_set}")
    print(f"{'='*60}")
    
    # Print fold results
    print("\nFold-wise Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: MSE: {result['mse']:.4f}, CI: {result['ci']:.4f}, RM2: {result['rm2']:.4f}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*60}")
    print(f"MSE - Mean: {statistics['mse']['mean']:.4f}, Std: {statistics['mse']['std']:.4f}, Var: {statistics['mse']['var']:.4f}")
    print(f"CI  - Mean: {statistics['ci']['mean']:.4f}, Std: {statistics['ci']['std']:.4f}, Var: {statistics['ci']['var']:.4f}")
    print(f"RM2 - Mean: {statistics['rm2']['mean']:.4f}, Std: {statistics['rm2']['std']:.4f}, Var: {statistics['rm2']['var']:.4f}")
    
    if all_test_predictions and all_test_labels:
        all_mse, all_ci, all_rm2 = calculate_metrics(np.array(all_test_labels), np.array(all_test_predictions))
        
        # Save overall metrics file
        overall_metrics_path = os.path.join(log_base_dir, f"overall_metrics_{hp.dataset}_{hp.running_set}_{hp.current_time}.csv")
        with open(overall_metrics_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mse", all_mse])
            writer.writerow(["ci", all_ci])
            writer.writerow(["rm2", all_rm2])

    print(f"\nComprehensive test log saved at: {test_log_path}")
    print(f"Training logs saved in: {log_base_dir}")
