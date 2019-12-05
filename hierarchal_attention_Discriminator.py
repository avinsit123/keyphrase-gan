#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:57:57 2019

@author: r17935avinash
"""


import torch 
from pykp.reward import *
from torch import nn as nn
import torch.nn.functional as F
import sys


def matmul(X, Y):
    taken = []
    for i in range(X.size(2)):
        result = (X[:,:,i]*Y)
        taken.append(result)
        results = torch.stack(taken,dim=2)
    return results

class S_RNN(nn.Module):
    def __init__(self,embed_dim,hidden_dim,n_layers,bidirectional=False):
        super(S_RNN,self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.RNN = nn.GRU(embed_dim,hidden_dim,n_layers)
        
        
    def forward(self,x):
        x = x.permute(1,0,2)
        x,hidden = self.RNN(x)
        x = x.permute(1,0,2)
        hidden = hidden.permute(1,0,2)
        return x,hidden
                  
class Discriminator(S_RNN):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_layers,pad_idx,devices,bidirectional=False):
        super().__init__(embedding_dim,hidden_dim,n_layers,bidirectional=False)

        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.devices = devices
        self.RNN1 = S_RNN(embedding_dim,hidden_dim,n_layers,True)  ### Abstract RNN
        self.RNN2 = S_RNN(embedding_dim,hidden_dim,n_layers,True)  ### Summary RNN
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.Compress = nn.Linear(2*hidden_dim,hidden_dim)
        self.attention = nn.Linear(hidden_dim,hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.MegaRNN = nn.GRU(hidden_dim,2*hidden_dim,n_layers)
        self.Linear = nn.Linear(2*hidden_dim,1)
        
        
        
    def padded_all(self,target,total_kphs,pad_id):
        #target =  [[1,2,3], [2,4,5,6], [2,4,6,7,8]]
        max_cols = max([len(row) for row in target])
        max_rows = total_kphs
        padded = [batch + [pad_id] * (max_rows - len(batch)) for batch in target]
        padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in padded])
        if torch.cuda.is_available():
            padded = padded.to(self.devices)
        return padded
        
    def transform_src(self,src,total_kphs):
        src = torch.Tensor(np.tile(src.cpu().numpy(),total_kphs))
        src = src.reshape(total_kphs,-1)
        if torch.cuda.is_available():
            src = src.to(self.devices)
        return src        
        
    def forward(self,src,kph):
        src = self.embedding(src)
        kph = self.embedding(kph)
        abstract_d = self.RNN1(src)[0]
        keyphrase_d = self.RNN2(kph)[1]
        keyphrase_d = keyphrase_d[:,0,:]
        abstract = abstract_d[0,:,:]
        return abstract,keyphrase_d
        

  
    def get_hidden_states(self,src,kph):
        total_kphs = len(kph)
        src = self.transform_src(src,total_kphs)
        kph = self.padded_all(kph,total_kphs,self.pad_idx)
        src,kph = src.long(),kph.long()    
        h_abstract,h_kph = self.forward(src,kph)
        h_abstract,h_kph = h_abstract.unsqueeze(0),h_kph.unsqueeze(0) 
        return h_abstract,h_kph
    
    def get_hidden_limit_length(self,src,kph,max_length):
        total_kphs = len(kph)
        src = src[:max_length]
        src = self.transform_src(src,total_kphs)
        kph = self.padded_all(kph,total_kphs,self.pad_idx)
        src,kph = src.long(),kph.long()    
        h_abstract,h_kph = self.forward(src,kph)
        h_abstract,h_kph = h_abstract.unsqueeze(0),h_kph.unsqueeze(0) 
        return h_abstract,h_kph        
    
    def calc_loss(self,output,target_type):
        total_len = output.size()
        if target_type==1:
            results = torch.ones(total_len)*0.9
            if torch.cuda.is_available():
                results = results.to(self.devices)
        else:
            results = torch.zeros(total_len)
            if torch.cuda.is_available():
                results = results.to(self.devices)
        criterion = nn.BCEWithLogitsLoss()
        avg_outputs = torch.mean(self.sigmoid(output))
        loss = criterion(output,results)
        return avg_outputs,loss
    
    def calc_rewards(self,output):      ## for maximizing f1 score 
        total_len = output.size()
        criterion = nn.BCEWithLogitsLoss()
        outputs = self.sigmoid(output)
        return outputs
    
    def calculate_context_rewards(self,abstract_t,kph_t,target_type,len_list):
        total_rewards = torch.Tensor([]).to(self.devices)
        total_rewards = total_rewards.unsqueeze(0)
        x = self.attention(abstract_t)
        temp = kph_t.permute(0,2,1)
        x = torch.bmm(x,temp)
        x = x.permute(0,2,1)
        x = torch.bmm(x,abstract_t)
        output = torch.cat((x,kph_t),dim=2)
        output = self.Linear(output)

        output = output.squeeze(2)
        total_reward,total_loss = 0,0
        for i,len_i in enumerate(len_list):
            r_output = output[i,:len_i].squeeze(0)
            reward = self.calc_rewards(r_output)
            reward = reward.unsqueeze(0)
            if(len(reward.size())==1):
                reward = reward.unsqueeze(0)
            total_rewards = torch.cat((total_rewards,reward),dim=1)
        total_rewards = total_rewards.squeeze(0)
        return total_rewards
        
    
    def calculate_context(self,abstract_t,kph_t,target_type,len_list):  ## for  maximizing f1 score
        x = self.attention(abstract_t)
        temp = kph_t.permute(0,2,1)
        x = torch.bmm(x,temp)
        x = x.permute(0,2,1)
        x = torch.bmm(x,abstract_t)
        output = torch.cat((x,kph_t),dim=2)
        output = self.Compress(output)
        output = output.squeeze(2)
        abstract_t = torch.mean(abstract_t,dim=1)
        abstract_t = abstract_t.unsqueeze(1)
        concat_output = torch.cat((abstract_t,kph_t),dim=1) 
        concat_output = concat_output.permute(1,0,2)
        x,hidden = self.MegaRNN(concat_output)
        x = x[-1,:,:]
        output = self.Linear(x)
        output = output.squeeze(1)
        total_len = output.size(0)
        if target_type==1:
            results = torch.ones(total_len)*0.9
            if torch.cuda.is_available():
                results = results.to(self.devices)
        else:
            results = torch.zeros(total_len)
            if torch.cuda.is_available():
                results = results.to(self.devices)
        criterion = nn.BCEWithLogitsLoss()
        avg_outputs = torch.mean(self.sigmoid(output))
        outputs = self.sigmoid(output)
        loss = criterion(output,results)
        return outputs,avg_outputs,loss
    
    def Catter(self,kph,rewards,total_len):
         lengths = [len(kp)+1 for kp in kph]
         max_len = max(lengths)
         x = torch.Tensor([])
         rewards_shape = rewards.repeat(max_len).reshape(-1,rewards.size(0)).t()
         x= torch.Tensor([])
         x = x.to(self.devices)
         for i,keyphrase in enumerate(rewards_shape):
             x = torch.cat((x,keyphrase[:lengths[i]]))
         x = F.pad(input=x,pad=(0,total_len-x.size(0)),mode='constant',value=0)
         return x    

    def calculate_rewards(self,abstract_t,kph_t,start_len,len_list,pred_str_list,total_len,gamma = 0.99):
        start_len = 1
        x = self.attention(abstract_t)
        temp = kph_t.permute(0,2,1)
        x = torch.bmm(x,temp)
        x = x.permute(0,2,1)
        x = torch.bmm(x,abstract_t)
        output = torch.cat((x,kph_t),dim=2)
        output = self.Compress(output)
        output = output.squeeze(2)
        abstract_t = torch.mean(abstract_t,dim=1)
        abstract_t = abstract_t.unsqueeze(1)
        concat_output = torch.cat((abstract_t,kph_t),dim=1) 
        concat_output = concat_output.permute(1,0,2)
        x,hidden = self.MegaRNN(concat_output)
        output = self.Linear(x)
        output = output.squeeze(2).t()
        avg_outputs = self.sigmoid(output)
        reward_outputs = torch.Tensor([]).to(self.devices)
        for i,len_i in enumerate(len_list):
            avg_outputs[i,start_len:start_len+len_i] = avg_outputs[i,start_len:start_len+len_i] 
            batch_rewards = self.Catter(pred_str_list[i],avg_outputs[i,start_len:start_len+len_i],total_len)
            reward_outputs = torch.cat((reward_outputs,batch_rewards))
        return reward_outputs
    
    def calculate_single_rewards(self,abstract_t,kph_t,start_len,len_list,pred_str_list,total_len,gamma = 0.99):
        x = self.attention(abstract_t)
        temp = kph_t.permute(0,2,1)
        x = torch.bmm(x,temp)
        x = x.permute(0,2,1)
        x = torch.bmm(x,abstract_t)
        output = torch.cat((x,kph_t),dim=2)
        output = self.Compress(output)
        output = output.squeeze(2)
        abstract_t = torch.mean(abstract_t,dim=1)
        abstract_t = abstract_t.unsqueeze(1)
        concat_output = torch.cat((abstract_t,kph_t),dim=1) 
        concat_output = concat_output.permute(1,0,2)
        x,hidden = self.MegaRNN(concat_output)
        x = x[-1,:,:]
        output = self.sigmoid(self.Linear(x))
        return output 
    
       
        
#D_model = Discriminator(50002,200,150,2,0) 



    
