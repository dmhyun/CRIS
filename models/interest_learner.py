import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class INTEREST_LEARNER(nn.Module):
    def __init__(self, opt):
        super(INTEREST_LEARNER, self).__init__()
        
        self.hd = opt.hidden_dim        
        
        self.proj = nn.Linear(self.hd*2,1)
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hd,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
    
    def forward(self, batch_data):
        frequency_bins = batch_data[1].float().cuda()
        
        output, _ = self.lstm(frequency_bins.unsqueeze(-1))
        interim = output[:,-1,:]
    
        prob = self.proj(interim)
        
        return nn.Sigmoid()(prob)       