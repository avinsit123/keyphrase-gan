#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:57:57 2019

@author: r17935avinash
"""


import torch 


#class Discriminator(nn.Module):

src = [2,4,5,6,3,1,4,2]
kph = [[2,4,2,4],[2,2,1],[3,1],[2,3,5,6]]


def padded_all(target,total_kphs,pad_id):
    #target =  [[1,2,3], [2,4,5,6], [2,4,6,7,8]]
    max_cols = max([len(row) for row in target])
    max_rows = total_kphs
    padded = [batch + [[pad_id] * (max_cols)] * (max_rows - len(batch)) for batch in target]
    padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in padded])
    padded = padded.view(-1, max_rows, max_cols)
    padded = padded[0]
    return padded
    
def transform_src(src,total_kphs):
    src = torch.Tensor(src * total_kphs)
    src = src.reshape(total_kphs,-1)
    return src
    
def train_one_abstract(src,kph,reward_type,batch_size,loss_criterion):
    #phrase_reward = np.zeros((batch_size, max_num_phrases))
    #phrase_reward = 
    total_kphs = len(kph)
    total_loss = 0
    src = transform_src(src)
    total_kphs = len(kph)
    kph = padded(kph,total_kphs,pad_id)
    output = self.forward(src,kph)
    loss = loss_criterion(output,reward_type)
    return loss
        
        #output = self.forward(src,kph)
        #loss = Loss(output,reward_type)
        #total_loss+=loss

    
def train_one_batch(src_str_list,pred_str_2dlist, trg_str_2dlist, batch_size):
    total_abstract_loss = 0
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src_str_list, pred_str_2dlist,trg_str_2dlist)):
        abstract_loss_real = train_one_abstract(src_list,pred_str_list,1,batch_size)
        abstract_loss_fake = train_one_abstract(src_list,pred_str_list,0,batch_size)
        total_abstract_loss+= abstract_loss_real + abstarct_loss_fake
    return (total_abstract_loss/batch_size)

def train_Discriminator(gen_model,train_data_loader,)
        
        
        
        
        
        
    


    