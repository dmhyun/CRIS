import os
import pdb
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from collections import Counter
from models import INTEREST_LEARNER
from torch.autograd import Variable
from sklearn.metrics import f1_score
from dataloaders.dataloader_interest import DataLoader

torch.set_num_threads(4)

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        self.data_loader = DataLoader(self.opt)
        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()

        trnlen = self.trn_loader.dataset.timediffs.shape[1]        

        print('TRN labels: {}'.format(Counter(self.trn_loader.dataset.labels)))
        print('VLD labels: {}'.format(Counter(self.vld_loader.dataset.labels)))
        print('TST labels: {}'.format(Counter(self.tst_loader.dataset.labels)))
        
        self.model = self.opt.model_class(self.opt).cuda()
        
        self._print_args()                
        
        
    def train(self):
        criterion = nn.BCELoss(reduction='none')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate)        
        
        best_vf1 = 0
        vld_f1 = 0
        tst_f1 = 0        
        batch_loss = 0        
        
        for epoch in range(self.opt.num_epoch):                        
            
            st = time.time()
            trn_outputs, trn_labels = [], []            
            for i, batch_data in enumerate(self.trn_loader):            
                batch_only_data = batch_data[:-1] # cuda will be called in the model
                labels = batch_data[-1].float().cuda()    
                
                if (labels>1).sum() != 0:
                    print('Label list contains an element not 0 or 1')
                    pdb.set_trace()
                
                class_weight = (labels == 1).float()                
                class_weight *= opt.pos_weight
                class_weight[class_weight==0] = (1-opt.pos_weight)
                class_weight = nn.functional.softmax(class_weight, dim=0)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_only_data).view(-1)
                loss = criterion(outputs, labels)
                
                loss = (loss * class_weight).sum()
                
                loss.backward()
                
                optimizer.step()

                batch_loss += loss.data.item()
                
                trn_outputs.append(outputs)
                trn_labels.append(labels)
        
            elapsed = time.time() - st

            evalt = time.time()
            
            trn_outputs = (torch.cat(trn_outputs) >= 0.5).float()    
            trn_labels = torch.cat(trn_labels)    
                        
            trn_f1 = f1_score(trn_labels.cpu().numpy(), trn_outputs.cpu().numpy(), average='binary')            
                        
            # Evaluation
            with torch.no_grad():
                
                vld_iids, vld_outputs, vld_labels = [], [], []
                for i, batch_data in enumerate(self.vld_loader):
                    batch_only_data = batch_data[:-1] # cuda will be called in models
                    
                    vld_iids += list(batch_data[0])                    
                    labels = batch_data[-1].float().cuda()

                    outputs = self.model(batch_only_data).view(-1)

                    vld_outputs.append(outputs)
                    vld_labels.append(labels)
                    
                vld_probs = torch.cat(vld_outputs)
                vld_outputs = (vld_probs >= 0.5).float()   
                vld_labels = torch.cat(vld_labels)    
                
                
                vld_f1 = f1_score(vld_labels.cpu().numpy(), vld_outputs.cpu().numpy(), average='binary')
                
    
                if vld_f1 > best_vf1:
                    best_vf1 = vld_f1

                    # Save ISSs of items
                    item_interest = np.vstack([np.array(vld_iids), vld_probs.cpu().numpy()])
                    recpath = '/'.join(opt.dataset_path.split('/')[:-1])+'/rec/'
                    if not os.path.exists(recpath): os.makedirs(recpath)
                    np.save(open(recpath+'/interest_prob', 'wb'), item_interest)
                    
                    tst_outputs, tst_labels = [], []
                    for k, batch_data in enumerate(self.tst_loader):
                        batch_only_data = batch_data[:-1]
                        labels = batch_data[-1].float().cuda()

                        outputs = self.model(batch_only_data).view(-1)

                        tst_outputs.append(outputs)
                        tst_labels.append(labels)

                    tst_outputs = (torch.cat(tst_outputs) >= 0.5).float()   
                    tst_labels = torch.cat(tst_labels)    

                    tst_f1 = f1_score(tst_labels.cpu().numpy(), tst_outputs.cpu().numpy(), average='binary')
                    
            evalt = time.time() - evalt
                    
            print(('(%.1fs, %.1fs)\tEpoch [%d/%d], trn_e : %5.4f, trn_f1 : %4.3f, vld_f1 : %4.3f, tst_f1 : %4.3f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), trn_f1, vld_f1,  tst_f1)))            
            
            batch_loss =0
                    
        print('VLD F1 and TST:\t{}\t{}'.format(best_vf1, tst_f1))
        
    
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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='intlearn', type=str)
    parser.add_argument('--dataset', default='tools', type=str)    
    parser.add_argument('--period', default=16, type=float)
    parser.add_argument('--binsize', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)    
    parser.add_argument('--hidden_dim', default=64, type=int)    
    parser.add_argument('--pos_weight', default=1e-2, type=float)   
    parser.add_argument('--gpu', default=3, type=int)       
    
    opt = parser.parse_args()
    
    torch.cuda.set_device(opt.gpu)
    
    model_classes = {
        'intlearn': INTEREST_LEARNER,      
    }
      
    dataset_path = './data/{}/split'.format(opt.dataset)    
    
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_path = dataset_path

    ins = Instructor(opt)
    ins.train()