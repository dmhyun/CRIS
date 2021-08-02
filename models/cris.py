import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class CRIS(nn.Module):
    def __init__(self, opt):
        super(CRIS, self).__init__()
        
        self.ebd_size = opt.K
        self.numuser = opt.numuser
        self.numitem = opt.numitem

        NUM_PROTOTYPE = 2 # prototype C and S
        
        self.ebd_user = nn.Embedding(self.numuser+1, self.ebd_size).cuda()
        self.ebd_item = nn.Embedding(self.numitem+1, self.ebd_size).cuda()    
        self.ebd_prototype = nn.Embedding(NUM_PROTOTYPE, self.ebd_size).cuda() 
        
        nn.init.xavier_normal_(self.ebd_user.weight)
        nn.init.xavier_normal_(self.ebd_item.weight)
        nn.init.xavier_normal_(self.ebd_prototype.weight)

        self.consumption_idx = torch.zeros(1).long().cuda()
        self.interest_idx = torch.ones(1).long().cuda()
        
    def forward(self, batch_data):
        user, item = batch_data
        
        consumption = self.ebd_prototype(self.consumption_idx)
        interest = self.ebd_prototype(self.interest_idx)
        
        embedded_user = self.ebd_user(user)
        embedded_item = self.ebd_item(item)
        
        ui_feature = embedded_user + embedded_item
        
        c_dist = F.pairwise_distance(consumption, ui_feature, 2)
        i_dist = F.pairwise_distance(interest, ui_feature, 2)
        
        return c_dist, i_dist


    
