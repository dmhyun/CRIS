import os
import pdb
import time
import math
import copy
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from models import CRIS
from numpy import std as STD
from numpy import average as AVG
from metric import cal_measures
from collections import Counter
from torch.autograd import Variable
from dataloaders.dataloader_recommendation import DataLoader

torch.set_num_threads(4)

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        self.data_loader = DataLoader(self.opt)

        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        
        opt.numuser = self.trn_loader.dataset.numuser
        opt.numitem = self.trn_loader.dataset.numitem
        self.model = self.opt.model_class(self.opt).cuda()
        
        self._print_args()
        
    def train(self):
        # Load ISSs of items
        iid, prob = np.load(opt.dataset_path+'/interest_prob')
        prob = prob.astype(float)

        try:
            itdict = np.load(opt.dataset_path+'/item_dict').item()
        except:
            itdict = np.load(opt.dataset_path+'/item_dict',allow_pickle=True).item() # for numpy 0.17+

        ipdict = {} # {item ID: its ISS}
        for i in range(len(iid)):
            itemid = iid[i]
            if itemid in itdict:
                ipdict[itdict[itemid]] = prob[i]
        
        newtime = round(time.time())        
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate)
            
        best_score = -1 
        best_topHits, best_topNdcgs = None, None
        batch_loss = 0
        c = 0 # to check early stopping
        
        self.clip_max_user = torch.FloatTensor([1.0]).cuda()
        self.clip_max_item = torch.FloatTensor([1.0]).cuda()
        self.clip_max_pro = torch.FloatTensor([1.0]).cuda()

        for epoch in range(self.opt.num_epoch):
            st = time.time()
    
            for i, batch_data in enumerate(self.trn_loader):
            
                # Unit-sphere restriction
                user_weight = self.model.ebd_user.weight.data
                user_weight.div_(torch.max(torch.norm(user_weight, 2, 1, True),
                                           self.clip_max_user).expand_as(user_weight))

                item_weight = self.model.ebd_item.weight.data
                item_weight.div_(torch.max(torch.norm(item_weight, 2, 1, True),
                                           self.clip_max_item).expand_as(item_weight))

                pro_weight = self.model.ebd_prototype.weight.data
                pro_weight.div_(torch.max(torch.norm(pro_weight, 2, 1, True),
                                           self.clip_max_pro).expand_as(pro_weight))  
                
                batch_data = [bd.cuda() for bd in batch_data]
                
                optimizer.zero_grad() 
                
                # Loss computation
                users, positems, negitems = batch_data

                c_posdist, i_posdist = self.model([users, positems])
                c_negdist, i_negdist = self.model([users, negitems])

                zero = torch.FloatTensor([0]).cuda()
                first_term = torch.max(c_posdist - c_negdist + opt.margin, zero)

                pp = [ipdict[it] for it in positems.tolist()]
                pn = [ipdict[it] for it in negitems.tolist()]

                pp = Variable(torch.FloatTensor(pp)).cuda()
                pn = Variable(torch.FloatTensor(pn)).cuda()

                second_term = torch.pow((i_posdist - i_negdist) - (pn - pp), 2)

                loss = first_term + opt.lamb * second_term

                loss = torch.mean(loss) 

                loss.backward()
                
                optimizer.step()
    
                batch_loss += loss.data.item()

            elapsed = time.time() - st
            evalt = time.time()
            
            with torch.no_grad():
                topHits, topNdcgs  = cal_measures(self.vld_loader, self.model, opt, ipdict)

                if (topHits[10] + topNdcgs[10])/2 > best_score:
                    best_score = (topHits[10] + topNdcgs[10])/2
                    
                    best_topHits = topHits
                    best_topNdcgs = topNdcgs
                    
                    c = 0
                    
                    test_topHits, test_topNdcgs = cal_measures(
                                    self.tst_loader, self.model, opt, ipdict)
                    
                evalt = time.time() - evalt 
            
            print(('(%.1fs, %.1fs)\tEpoch [%d/%d], TRN_ERR : %.4f, v_score : %5.4f, tHR@10 : %5.4f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), (topHits[10] + topNdcgs[10])/2,  test_topHits[10])))

            batch_loss = 0
            
            c += 1
            if c > 5: break # Early-stopping
        
        print(('\nValid score@10 : %5.4f, HR@10 : %5.4f, NDCG@10 : %5.4f\n'% (((best_topHits[10] + best_topNdcgs[10])/2), best_topHits[10],  best_topNdcgs[10])))
        
        return test_topHits,  test_topNdcgs
            
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('\nn_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        print('')

    def run(self, repeats):
        results = []
        rndseed = [19427, 78036, 37498, 87299, 60330] # randomly-generated seeds
        for i in range(repeats):
            print('\n💫 repeat: {}/{}'.format(i+1, repeats))
            random.seed(rndseed[i]); np.random.seed(rndseed[i]); torch.manual_seed(rndseed[i])
            self._reset_params()
            
            results.append(ins.train())
        
        results = np.array(results)
        
        hrs_mean = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_mean = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        
        hrs_std = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_std = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        
    
        print('*TST Performance\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_mean.astype(str))))
        print('*NDCG means: {}'.format(', '.join(ndcg_mean.astype(str))))
        
    def _reset_params(self):
        self.model = self.opt.model_class(self.opt).cuda()
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='cris', type=str)
    parser.add_argument('--dataset', default='tools', type=str)    
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)    
    parser.add_argument('--batch_size', default=4096, type=int)    
    parser.add_argument('--margin', default=0.6, type=float)
    parser.add_argument('--lamb', default=0.2, type=float)
    parser.add_argument('--gamma', default=1.6, type=float)
    parser.add_argument('--K', default=50, type=int)      
    parser.add_argument('--numneg', default=10, type=int)
    parser.add_argument('--gpu', default=3, type=int)
    
    opt = parser.parse_args()
    
    torch.cuda.set_device(opt.gpu)
    
    model_classes = {        
        'cris':CRIS,        
    }  
    
    dataset_path = './data/{}/rec'.format(opt.dataset)
    
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_path = dataset_path

    ins = Instructor(opt)
    
    ins.run(5)     
