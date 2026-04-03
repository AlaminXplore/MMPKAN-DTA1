class HyperParameter:
    def __init__(self):
        self.kfold = 5
        self.data_root = './MMPKANDTA/datasets/'
        self.dataset = 'kiba'
        self.running_set = 'warm'
        
        self.mol2vec_dir = f'./MMPKANDTA/datasets/pretrained/{self.dataset}_chem_pretrained.pkl'
        self.protvec_dir = f'./MMPKANDTA/datasets/pretrained/{self.dataset}_esmc_pretrain.pkl'
        self.contact_map = f'./MMPKANDTA/datasets/pretrained/{self.dataset}_esm2_contact_map.pkl'        
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv' 

        self.Learning_rate = 1e-4
        self.Epoch = 300
        self.Batch_size = 16
        self.max_patience = 20

        self.drug_max_len = 220
        self.prot_max_len = 1200
        self.mol2vec_dim = 384
        self.protvec_dim = 1152

        self.cuda = '1'
        self.dropout = 0.2

    def set_dataset(self, data_name):
        self.dataset = data_name
        self.mol2vec_dir = f'./MMPKANDTA/datasets/pretrained/{self.dataset}_chem_pretrained.pkl'
        self.protvec_dir = f'./MMPKANDTA/datasets/pretrained/{self.dataset}_esmc_pretrain.pkl'
        self.contact_map = f'./MMPKANDTA/datasets/pretrained/{self.dataset}_esm2_contact_map.pkl'           
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'
