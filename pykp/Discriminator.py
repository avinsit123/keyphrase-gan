#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:57:57 2019

@author: r17935avinash
"""


import torch 
from pykp.reward import *
from torch import nn as nn

#class Discriminator(nn.Module):
"""
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
    src = transform_src(src)
    total_kphs = len(kph)
    kph = padded_all(kph,total_kphs,self.pad_idx)
    output = self.forward(src,kph)
    loss = loss_criterion(output,reward_type)
    return loss
        
        #output = self.forward(src,kph)
        #loss = Loss(output,reward_type)
        #total_loss+=loss

    
def train_one_batchs(src_str_list,pred_str_2dlist, trg_str_2dlist, batch_size):
    total_abstract_loss = 0
    
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src_str_list, pred_str_2dlist,trg_str_2dlist)):
        abstract_loss_real = train_one_abstract(src_list,pred_str_list,1,batch_size)
        abstract_loss_fake = train_one_abstract(src_list,pred_str_list,0,batch_size)
        total_abstract_loss+= abstract_loss_real + abstarct_loss_fake
    return (total_abstract_loss/batch_size)


def train_one_batch(one2many_batch, generator, optimizer):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = one2many_batch
    src = src.to(self.device)
    src_mask = src_mask.to(self.device)
    src_oov = src_oov.to(self.device)   
    eos_idx = self.eos_idx
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                              src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    total_abstract_loss = 0
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src_str_list, pred_str_2dlist,trg_str_2dlist)):
        abstract_loss_real = train_one_abstract(src_list,pred_str_list,1,batch_size)
        abstract_loss_fake = train_one_abstract(src_list,pred_str_list,0,batch_size)
        total_abstract_loss+= abstract_loss_real + abstarct_loss_fake
    avg_batch_loss = (total_abstract_loss/batch_size)
   
    avg_batch_loss.backward()
    optimizer.step()
   
    return avg_batch_loss.detach().item()

""" 
   
class SRNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,n_layers,bidirectional=False):
        super(SRNN,self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.RNN = nn.GRU(embed_dim,hidden_dim,n_layers)
        
        
    def forward(self,x):
        x = self.embedding(x)
        #print(x.size())
        x = x.permute(1,0,2)
        x,hidden = self.RNN(x)
        x = x.permute(1,0,2)
        hidden = hidden.permute(1,0,2)
        return x,hidden
        
def padded_all(target,total_kphs,pad_id):
    #target =  [[1,2,3], [2,4,5,6], [2,4,6,7,8]]
    max_cols = max([len(row) for row in target])
    max_rows = total_kphs
    padded = [batch + [pad_id] * (max_rows - len(batch)) for batch in target]
    padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in padded])
    padded = padded.view(-1, max_rows, max_cols)
    padded = padded[0]
    return padded
    
def transform_src(src,total_kphs):
    src = torch.Tensor(src * total_kphs)
    src = src.reshape(total_kphs,-1)
    return src


 #   fake_labels = torch.from_numpy(np.random.uniform(0, 0.3, size=(BATCH_SIZE))).float().to(DEVICE)
 #   real_labels = torch.from_numpy(np.random.uniform(0.7, 1.2, size=(BATCH_SIZE))).float().to(DEVICE)

        

def calculate_loss(output,target_type):
    total_len = output.size(0)
    if target_type==1:
        results = torch.ones(total_len)*0.9
    else:
        results = torch.zeros(total_len)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output,results)
    return loss
        
        
class Discriminator(nn.Module):
    def __init__(self,src_RNN,tgt_RNN,one2many,bos_idx,eos_idx,pad_idx,peos_idx,max_len,copy_attention,review_attn,coverage_attn,epochs,device,embedding_dim):
        super(Discriminator,self).__init__()
        
        
        #### INDICES INITIALIZATION
        
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.peos_idx = peos_idx
        self.max_len = max_len
        self.epoch = epochs
        self.one2many = one2many
        
        ##### MODEL SPECIFICATIONS
        self.RNN1 = src_RNN
        self.RNN2 = tgt_RNN
        
        
        
        
        
        
        
    def forward(src,kph):
        s_len = src.size(0)
        
        abstract_d = self.RNN1(src)[0]
        abstract_d = abstract_d.permute(1,0,2)
        keyphrase_d = self.RNN2(kph)[1]
        keyphrase_d = keyphrase_d[:,0,:]
        keyphrase_d = keyphrase_d.unsqueeze(2)
        
        ### Cosine Filter
        cosine_results = torch.bmm(abstract_d,keyphrase_d).squeeze(2)  ## Size(n_kphs,max_len)
        cosine_avgs = torch.mean(cosine_results,dim=1)
        return cosine_avgs
     

        
            
            
            
    
        
        
    def train_one_abstract(self,src,kph,reward_type,batch_size,loss_criterion):
        #phrase_reward = np.zeros((batch_size, max_num_phrases))
        #phrase_reward = 
        total_kphs = len(kph)
        src = transform_src(src,total_kphs)
        #total_kphs = len(kph)
        kph = padded_all(kph,total_kphs,self.pad_idx)
        src,kph = src.long(),kph.long()
        output = self.forward(src,kph)
        total_len = output.size(0)
        if target_type==1:
            results = torch.ones(total_len)*0.9
        else:
            results = torch.zeros(total_len)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output,results)
        return loss 
    
    def train_one_batch(self,one2many_batch, generator, optimizer):
        optimizer.zero_grad()
        
        src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = one2many_batch
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        src_oov = src_oov.to(self.device)   
        eos_idx = self.eos_idx
        sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
            src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
            one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
        pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                  src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
        total_abstract_loss = 0
        for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src_str_list, pred_str_2dlist,trg_str_2dlist)):
            abstract_loss_real = train_one_abstract(src_list,pred_str_list,1,batch_size)
            abstract_loss_fake = train_one_abstract(src_list,pred_str_list,0,batch_size)
            total_abstract_loss+= abstract_loss_real + abstarct_loss_fake
        avg_batch_loss = (total_abstract_loss/batch_size)
       
        avg_batch_loss.backward()
        optimizer.step()
        
        return avg_batch_loss.detach().item()    
        
        
        
    def train_Discriminator(self,model,optimizer_rl,train_loader,valid_loader):
        
            generator = SequenceGenerator(model,bos_idx=opt.word2idx[self.bos_index],
                                  eos_idx=opt.word2idx[self.eos_idx],
                                  pad_idx=opt.word2idx[self.pad_idx],
                                  peos_idx=opt.word2idx[self.peos_idx],
                                  beam_size=1,
                                  max_sequence_length=self.max_length,
                                  copy_attn=self.copy_attention,
                                  coverage_attn=self.coverage_attn,
                                  review_attn=self.review_attn,
                                  cuda=self.device_id > -1 
                                  )        
        for epoch in range(self.epoch):

            total_batch = 0
            
            generator.model.train()
            for i,batch in enumerate(train_loader):
                total_batch+=1
                batch_loss = train_one_batch(batch, generator, optimizer_rl)
                
                
#src_RNN = SRNN()
# optimizer_rl = torch.optim.Adam(list(src_RNN.parameters())+list(tgt_RNN.parameters()),lr=0.01)    
        
""" TEST_ASD
 keywords = [[2,3,4],[3,2],[4,5,3,4,2]]
 summary = [1,2,3,4,5,6]
 total_kphs = len(kph)
 
 src_RNN = SRNN(50000,200,150,2)
 tgt_RNN = SRNN(5000,200,150,2)
 tgt_RNN(kph.long())[0].size()
 hidden_d = tgt_RNN(kph.long())[1].size()
 keyphrase_d = hidden_d[:,0,:]
 keyphrase_d = keyphrase_d.squeeze(2)
 abstract_d = abstract_d.permute(1,0,2)
 cosine_results = torch.bmm(abstract_d,keyphrase_d).squeeze(2)  
 cosine_avgs = torch.mean(cosine_results,dim=1)
 
 cosine_results = torch.bmm(abstract_d,keyphrase_d).squeeze(2)  ## Size(n_kphs,max_len)
 cosine_avgs = torch.mean(cosine_results,dim=1)
 lossd = calculate_loss(cosine_avdgs,1) + calculate_loss(cosine_avdgs,0)
 
"""
        
    


    