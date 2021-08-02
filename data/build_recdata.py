import os
import sys
import csv
import pdb
import copy
import random
import numpy as np
import itertools
from collections import Counter

def replace_id2idx(trn, vld, tst):
    
    def build_dict(category):
        category = list(set(category))

        cate_dict = {}
        for i, c in enumerate(category): cate_dict[c] = i
        return cate_dict

    def id2idx(uir, udict, idict): # Convert IDs in string into IDs in numbers
        newuir = []
        for i in range(len(uir)):
            user, item, rating, _ = uir[i] # Fourth element is a time stamp for the interaction
            newuir.append([udict[user], idict[item], rating])
        return newuir

    trn_users = [i[0] for i in trn] 
    trn_items = [i[1] for i in trn] 
    
    user_dict = build_dict(trn_users)
    item_dict = build_dict(trn_items)
    
    trn = id2idx(trn, user_dict, item_dict)
    vld = id2idx(vld, user_dict, item_dict)
    tst = id2idx(tst, user_dict, item_dict)
    
    return trn, vld, tst, user_dict, item_dict

def load_raw_data(fn):
    print('Load ' + fn)
    rawdata = [l for l in csv.reader(open(fn))]
    return rawdata

def find_negatives(dataset):
    NUMNEG = 100
    
    trn, vld, tst = dataset
    
    allitems = set([i[1] for i in trn])
    
    uidict = {} # {u: [items consumed by user u]}
    for i in range(len(trn)):
        user, item, rating = trn[i]
        if user not in uidict: uidict[user] = []
        uidict[user].append(item)
    
    for i in range(len(vld)):
        user, item, _ = vld[i]
            
        useritems = set(uidict[user] + [item]) # Target item and a user's consumed items
        negative_items = random.sample(list(allitems - useritems), NUMNEG)
        
        vld[i] = vld[i][:-1] + negative_items # Append negative items for evaluation
    
    for i in range(len(tst)):
        user, item, _ = tst[i]
        
        useritems = set(uidict[user] + [item])
        negative_items = random.sample(list(allitems - useritems), NUMNEG) 
        
        tst[i] = tst[i][:-1] + negative_items
    
    return trn, vld, tst
    

dn = sys.argv[1] + '/' if not sys.argv[1].endswith('/') else sys.argv[1]
data_path = dn+'split/' if 'split/' not in dn else dn

print('\n🧰 Building a dataset for training the recommender system \n')

for fn in os.listdir(data_path):
    if 'trn' in fn: trndata_name = data_path+fn
    if 'vld' in fn: vlddata_name = data_path+fn
    if 'tst' in fn: tstdata_name = data_path+fn

# Load datasets and review features from csv format
trndata = load_raw_data(trndata_name)
vlddata = load_raw_data(vlddata_name)
tstdata = load_raw_data(tstdata_name)

trndata, org_vlddata, org_tstdata, user2id_dict, item2id_dict = replace_id2idx(trndata, vlddata, tstdata)

trndat, vlddata, tstdata = find_negatives([trndata, copy.deepcopy(org_vlddata), copy.deepcopy(org_tstdata)])

print('\nTRN:{}\tVLD:{}\tTST:{}'.format(len(trndata), len(vlddata), len(tstdata)))

print('\n📂 Starting to save datasets')
dataset_name = dn.split('/')[0] 
base_path = dataset_name+'/rec/'
if not os.path.exists(base_path): os.makedirs(base_path)

data_path = base_path

np.save(open(data_path+'trn','wb'), np.array(trndata).astype(float).astype(int))
np.save(open(data_path+'vld','wb'), np.array(vlddata).astype(float).astype(int))
np.save(open(data_path+'tst','wb'), np.array(tstdata).astype(float).astype(int))
np.save(open(data_path+'user_dict','wb'), user2id_dict)
np.save(open(data_path+'item_dict','wb'), item2id_dict)

print('\nDatasets saved to the data directory: {}\n'.format(data_path))
