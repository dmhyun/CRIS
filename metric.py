import pdb
import math
import torch
import random
import numpy as np

def HitRatio(ranklist):
    return bool(1 in ranklist)    

def NDCG(ranklist):
    for i, label in enumerate(ranklist):
        if label == 1: # True consumption
            return math.log(2) / math.log(i+2)
    return 0

def return_perf(predictions, ipdict):
    predictions = predictions.reshape(-1, 101, 4) # 1 positive and 100 negatives
        
    topks = [2,5,10,20]
    
    hrs, ndcgs = {}, {}
    for tk in topks:
        hrs[tk] = 0
        ndcgs[tk] = 0
    
    for row in predictions: 
        inst = row[:, 1:] # [i, score, label]
        # To set wrong if all predictions are the same,
        # move the positive item to the end of the list.
        inst[[0, -1]] = inst[[-1, 0]] 
        inst = inst[inst[:,1].argsort()] # items with small distance will be at upper position
        
        for tk in topks:
            topk_labels = inst[:tk, -1]
            hrs[tk] += HitRatio(topk_labels)
            ndcgs[tk] += NDCG(topk_labels)
    
    numinst = predictions.shape[0]
    
    for tk in topks:
        hrs[tk] /= numinst
        ndcgs[tk] /= numinst
        
    return hrs, ndcgs
        
def _cal_ranking_measures(loader, model, opt, ipdict):
    predictions = np.array([])
    all_output, all_label = [], []
    all_uid, all_iid = [], []  
    all_interest = []
    
    for i, batch_data in enumerate(loader):
        batch_data = [bd.cuda() for bd in batch_data]
        
        user, item, label = batch_data
        dist, interest = model([user, item])

        all_interest.append(interest)
            
        all_output.append(dist)
        all_label.append(label)
        all_uid.append(user)
        all_iid.append(item)

    all_output = torch.cat(all_output).cpu().data.numpy()
    all_label = torch.cat(all_label).cpu().data.numpy()
    all_uid = torch.cat(all_uid).cpu().data.numpy()
    all_iid = torch.cat(all_iid).cpu().data.numpy()    
    
    if len(all_interest) != 0: all_interest = torch.cat(all_interest).cpu().data.numpy()
        
    total_output = all_output + opt.gamma * all_interest        
    predictions = np.array([all_uid, all_iid, total_output, all_label]).T
    hrs, ndcgs = return_perf(predictions, ipdict)
    
    return hrs, ndcgs

def cal_measures(loader, model, opt, ipdict):
    model.eval()    
    
    results = _cal_ranking_measures(loader, model, opt, ipdict)
    
    model.train()
    
    return results