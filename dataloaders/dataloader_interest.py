import os
import csv
import pdb
import time
import pickle
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from torch.utils import data
from torch.utils.data.dataloader import default_collate

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

def toymd(time):
    return datetime.utcfromtimestamp(time)#.strftime('%Y-%m-%d')

class Dataset(data.Dataset):

    def __init__(self, data):
        st = time.time()
        
        self.iids, self.labels, self.timediffs = [], [], []
        self.most_oldtime = None
        
        for row in data:
            self.iids.append(row[0])
            self.labels.append(row[1])
            self.timediffs.append(row[2:])
            
        self.iids = np.array(self.iids)
        self.timediffs = np.array(self.timediffs).astype(int)
        self.labels = (np.array(self.labels) == 'True').astype(int) 
        
        print('Data building time : %.1fs' % (time.time()-st))
        
    def __getitem__(self, index):
        return self.iids[index], self.timediffs[index], self.labels[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.timediffs)
    
def build_loader(eachdata, batch_size, shuffle=True, num_workers=0):
    
    def my_collate(batch):
        batch = [i for i in filter(lambda x:x is not None, batch)]
        return default_collate(batch)
    
    """Builds and returns Dataloader."""
    dataset = Dataset(eachdata)
    
    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)

    return data_loader    

def build_data_directly(dpath, period, binsize):
    def toymd(time):
        return datetime.utcfromtimestamp(time)
    
    def build_data(true_items, item_feature):
        output = []
        for i in item_feature:
            feature = item_feature[i]
            instance = [i] + [bool(i in true_items)] + list(feature) # [iid, label, features]
            output.append(instance)    
        return np.array(output)
    
    def get_item_feature(data):
        times = data[:,-1].astype(float).astype(int)
        mintime, maxtime = toymd(min(times)), toymd(max(times))

        # Binning training time (D_f) with fixed-sized bins
        timedelta = relativedelta(weeks=binsize)
        bins = np.array([mintime + timedelta*i for i in range(1000) # quick implementation
                         if mintime + timedelta*i < maxtime + timedelta*0])

        # Build features from data
        idict = {}
        for u, i, r, t in data:
            if i not in idict: idict[i] = []
            idict[i].append(toymd(int(float(t))))

        # Build features for each item
        item_feature = {}
        for i in idict:
            times = np.array(idict[i])

            # Transform times into frequency bins
            binned_times = []
            for t in times:
                binidx = np.where(bins <= t)[0][-1]
                each_binfeature = np.zeros(len(bins))
                each_binfeature[binidx] = 1
                binned_times.append(each_binfeature)
            binned_times = np.array(binned_times).sum(axis=0).astype(int)

            item_feature[i] = binned_times
            
        return item_feature

    rawtrn = np.array([l for l in csv.reader(open(dpath+'trn.csv'))])
    rawvld = np.array([l for l in csv.reader(open(dpath+'vld.csv'))])
    rawtst = np.array([l for l in csv.reader(open(dpath+'tst.csv'))])
    
    times_trn = rawtrn[:,-1].astype(int)
    
    # Split data by period (unit: week)
    # [trn_start - trnfront - vld_start - tst_start - tst_end]
    trnfront_time = times_trn.max() - 60 * 60 * 24 * 7 * period 
    trnfront_idx = np.where(times_trn < trnfront_time)[0][-1]
    trn_start_time = int(float(times_trn[0])) # -1 denotes the time index
    trnfront_start_time = int(float(rawtrn[trnfront_idx][-1]))
    vld_start_time = int(float(rawvld[0][-1]))
    tst_start_time = int(float(rawtst[0][-1]))
    tst_end_time = int(float(rawtst[-1][-1]))
    
    print('\n📋 Data loaded from: {}\n'.format(dpath))

    print('Trn start time:\t{}'.format(toymd(trn_start_time)))
    print('Trn front time:\t{}'.format(toymd(trnfront_start_time)))
    print('Vld start time:\t{}'.format(toymd(vld_start_time)))
    print('Tst start time:\t{}'.format(toymd(tst_start_time)))
    print('Tst end time:\t{}'.format(toymd(tst_end_time)))
    
    trn_4feature = rawtrn[:trnfront_idx]
    feature_trn = get_item_feature(trn_4feature) # features for training
    feature_eval = get_item_feature(rawtrn) # features for evaluation (to get ISS for training RS)
    
    trn_4label = rawtrn[trnfront_idx:] # D_b
    
    trndata = build_data(set(trn_4label[:,1]), feature_trn)
    vlddata = build_data(set(rawvld[:,1]), feature_eval)
    tstdata = build_data(set(rawtst[:,1]), feature_eval)
    
    return trndata, vlddata, tstdata

class DataLoader:
    def __init__(self, opt):
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        
        trndata, vlddata, tstdata = build_data_directly(self.dpath, opt.period, opt.binsize)        
        
        self.trn_loader = build_loader(trndata, opt.batch_size, shuffle=True)
        self.vld_loader = build_loader(vlddata, opt.batch_size, shuffle=False)
        self.tst_loader = build_loader(tstdata, opt.batch_size, shuffle=False)
        
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("==================================================================================")
            
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding