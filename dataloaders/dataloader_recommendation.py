import os
import pdb
import time
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data.dataloader import default_collate

random.seed(2020)

class ML_Dataset(data.Dataset):
    
    def build_consumption_history(self, uir):
        # Build a dictionary for user: items consumed by the user
        uir = uir.astype(int)
        uidict = {}
        allitems = set()
        for u, i, _ in uir:
            if u not in uidict: uidict[u] = set()
            uidict[u].add(i)
            allitems.add(i)
            
        self.ui_cand_dict = {}    
        for u in uidict:
            self.ui_cand_dict[u] = np.array(list(allitems - uidict[u]))
        
        return uidict, allitems
        
    def __init__(self, path, trn_numneg):
        dpath = '/'.join(path.split('/')[:-1])
        if dpath[-1] != '/': dpath += '/'
        dtype = path.split('/')[-1]
        
        st = time.time()        
        
        if dtype == 'trn': 
            self.numneg = trn_numneg
            trn = np.load(dpath+'trn')
            self.uir = trn
        elif dtype == 'vld': self.uir = np.load(dpath+'vld')
        elif dtype == 'tst': self.uir = np.load(dpath+'tst')
            
        if dtype == 'trn':             
            self.uir[:,-1] = 1 # Mark explicit feedback as implicit feedback

            self.first = self.uir[:,0].astype(int)
            self.second = self.uir[:,1].astype(int)
            self.third = np.zeros(self.uir.shape[0]) # This will be replaced in 'train_collate'
            
            self.numuser = len(set(self.uir[:,0].astype(int)))
            self.numitem = len(set(self.uir[:,1].astype(int)))
            
            self.uidict, self.allitems = self.build_consumption_history(self.uir)
            
        elif dtype == 'vld' or dtype == 'tst':             
            # Build validation data for ranking evaluation
            newuir = []
            for row in self.uir:
                user = row[0]
                true_item = row[1]
                newuir.append([user, true_item, 1]) # a true consumption
                for item in row[2:]: newuir.append([user, item, 0]) # negative candidates
            self.uir = np.array(newuir) # User, Item, Rating
        
            self.first, self.second, self.third = self.uir[:,0], self.uir[:,1], self.uir[:,2]
        
        
        print('Data building time : %.1fs' % (time.time()-st))

    def __getitem__(self, index):
        # Training: [user, positive, negative]
        # Testing: [user, canidate item, label] 
        return self.first[index], self.second[index], self.third[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.first)
    
    
    def train_collate(self, batch):
        # Input: [user, postive item, dummy]
        # Output: [user, positive item, negative item]
        batch = [i for i in filter(lambda x:x is not None, batch)]
        
        # Negative sampling for each batch
        outputs = []
        for u, pi, dummy in batch:
            rand_idx = np.random.randint(len(self.ui_cand_dict[u]), size=self.numneg)
            neg_items = self.ui_cand_dict[u][rand_idx]
            
            for ni in neg_items: 
                outputs.append([u, pi, ni])
            
        return default_collate(outputs)      
    
def test_collate(batch):
    batch = [i for i in filter(lambda x:x is not None, batch)]
    return default_collate(batch)

def get_each_loader(data_path, batch_size, trn_negnum, shuffle=True, num_workers=0):
    """Builds and returns Dataloader."""
    
    dataset = ML_Dataset(data_path, trn_negnum)
    
    if data_path.endswith('trn') == True:
        collate = dataset.train_collate
    else:
        collate = test_collate

    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate)

    return data_loader

class DataLoader: 
    def __init__(self, opt):
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        self.trn_numneg = opt.numneg
        
        self.trn_loader, self.vld_loader, self.tst_loader = self.get_loaders_for_metric_learning(self.trn_numneg)
    
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("=" * 80)
        
    def get_loaders_for_metric_learning(self, trn_numneg):
        print("\n📋 Loading data...\n")
        trn_loader = get_each_loader(self.dpath+'trn', self.batch_size, trn_numneg, shuffle=True)
        print('\tTraining data loaded')
        
        vld_loader = get_each_loader(self.dpath+'vld', self.batch_size, trn_numneg, shuffle=False)
        print('\tValidation data loaded')
        
        tst_loader = get_each_loader(self.dpath+'tst', self.batch_size, trn_numneg, shuffle=False)
        print('\tTest data loaded')
        
        return trn_loader, vld_loader, tst_loader
    
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
            
