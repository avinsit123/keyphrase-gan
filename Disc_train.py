#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:10:45 2019

@author: r17935avinash
"""

################################ IMPORT LIBRARIES ###############################################################
import torch
import numpy as np
import pykp.io
import torch.nn as nn
from utils.statistics import RewardStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_train_and_valid_loss, export_train_and_valid_reward
import sys
import logging
import os
from evaluate import evaluate_reward
from pykp.reward import *
import math
EPS = 1e-8
import argparse
import config
import logging
import os
import json
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqModel
from torch.optim import Adam
import pykp
from pykp.model import Seq2SeqModel
import train_ml
import train_rl

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
from utils.string_helper import convert_list_to_kphs
import time
import numpy as np
import random
from torch import device 
from hierarchal_attention_Discriminator import Discriminator
from torch.nn import functional as F
#####################################################################################################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
#def Check_Valid_Loss(valid_data_loader,D_model,batch,generator,opt,perturb_std):
    
##### TUNE HYPERPARAMETERS ##############

##  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################
def train_one_batch(D_model,one2many_batch, generator, opt,perturb_std):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = one2many_batch
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1
    if torch.cuda.is_available():
        src = src.to(opt.device)
        src_mask = src_mask.to(opt.device)
        src_oov = src_oov.to(opt.device)
    if opt.title_guided:
        title = title.to(opt.device)
        title_mask = title_mask.to(opt.device)
        
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type
    reward_shaping = opt.reward_shaping
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_type = opt.regularization_type ## DNT
    regularization_factor = opt.regularization_factor ##DNT
    devices = opt.device
    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False
    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    
    
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                              src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
     
    target_str_2dlist = convert_list_to_kphs(trg)
    """
     src = [batch_size,abstract_seq_len]
     target_str_2dlist = list of list of true keyphrases
     pred_str_2dlist = list of list of false keyphrases
    
    """
    total_abstract_loss = 0
    batch_mine = 0
    abstract_t = torch.Tensor([]).to(devices)
    abstract_f = torch.Tensor([]).to(devices)
    kph_t = torch.Tensor([]).to(devices)
    kph_f = torch.Tensor([]).to(devices)    
    h_kph_t_size = 0
    h_kph_f_size = 0 
    len_list_t,len_list_f = [],[]
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src, pred_str_2dlist,target_str_2dlist)):
        batch_mine+=1
        if (len(target_str_list)==0 or len(pred_str_list)==0):
            continue
        h_abstract_t,h_kph_t = D_model.get_hidden_states(src_list,target_str_list)
        h_abstract_f,h_kph_f = D_model.get_hidden_states(src_list,pred_str_list)
        len_list_t.append(h_kph_t.size(1))
        len_list_f.append(h_kph_f.size(1))
        h_kph_t_size = max(h_kph_t_size,h_kph_t.size(1))
        h_kph_f_size = max(h_kph_f_size,h_kph_f.size(1))
    
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src, pred_str_2dlist,target_str_2dlist)):
        batch_mine+=1
        if (len(target_str_list)==0 or len(pred_str_list)==0):
            continue  
        h_abstract_t,h_kph_t = D_model.get_hidden_states(src_list,target_str_list)
        h_abstract_f,h_kph_f = D_model.get_hidden_states(src_list,pred_str_list)
        p1d = (0,0,0,h_kph_t_size - h_kph_t.size(1))
        p2d = (0,0,0,h_kph_f_size - h_kph_f.size(1))
        h_kph_t = F.pad(h_kph_t,p1d)
        h_kph_f = F.pad(h_kph_f,p2d)
        abstract_t = torch.cat((abstract_t,h_abstract_t),dim=0)
        abstract_f = torch.cat((abstract_f,h_abstract_f),dim=0)
        kph_t = torch.cat((kph_t,h_kph_t),dim=0)
        kph_f = torch.cat((kph_f,h_kph_f),dim=0)
    _,real_rewards,abstract_loss_real = D_model.calculate_context(abstract_t,kph_t,1,len_list_t)  
    _,fake_rewards,abstract_loss_fake = D_model.calculate_context(abstract_f,kph_f,0,len_list_f)  
    avg_batch_loss = ( abstract_loss_real + abstract_loss_fake )
    avg_real = real_rewards
    avg_fake = fake_rewards
    return avg_batch_loss,avg_real,avg_fake

def main(opt):
    clip = 5
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)
    
    print("Data Successfully Loaded __.__.__.__.__.__.__.__.__.__.__.__.__.__.")
    model = Seq2SeqModel(opt)
    
##    if torch.cuda.is_available():
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(opt.model_path))
        model = model.to(opt.gpuid)
    else:
        model.load_state_dict(torch.load(opt.model_path,map_location="cpu"))
    
    print("___________________ Generator Initialised and Loaded _________________________")
    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  peos_idx=opt.word2idx[pykp.io.PEOS_WORD],
                                  beam_size=1,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  cuda=opt.gpuid > -1
                                  )
    
    init_perturb_std = opt.init_perturb_std
    final_perturb_std = opt.final_perturb_std
    perturb_decay_factor = opt.perturb_decay_factor
    perturb_decay_mode = opt.perturb_decay_mode
    hidden_dim = opt.D_hidden_dim
    embedding_dim = opt.D_embedding_dim 
    n_layers = opt.D_layers
    D_model = Discriminator(opt.vocab_size,embedding_dim,hidden_dim,n_layers,opt.word2idx[pykp.io.PAD_WORD])
    print("The Discriminator Description is ",D_model)
    
    if opt.pretrained_Discriminator :
        if torch.cuda.is_available() :
            D_model.load_state_dict(torch.load(opt.Discriminator_model_path))
            D_model = D_model.to(opt.gpuid)
        else:
            D_model.load_state_dict(torch.load(opt.Discriminator_model_path,map_location="cpu"))
    else :
        if torch.cuda.is_available() :
            D_model = D_model.to(opt.gpuid)
        else:
            D_model.load_state_dict(torch.load(opt.Discriminator_model_path,map_location="cpu"))        
    D_optimizer = torch.optim.Adam(D_model.parameters(),opt.learning_rate)
    print("Beginning with training Discriminator")
    print("########################################################################################################")
    total_epochs = 5
    for epoch in range(total_epochs):
        total_batch = 0
        print("Starting with epoch:",epoch)
        for batch_i, batch in enumerate(train_data_loader):
            best_valid_loss = 1000
            D_model.train()
            D_optimizer.zero_grad()
            
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(-1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1+total_batch)/4000))
            avg_batch_loss , _ , _ = train_one_batch( D_model , batch , generator , opt , perturb_std )
            torch.nn.utils.clip_grad_norm_( D_model.parameters() , clip)
            avg_batch_loss.backward()
            
            D_optimizer.step()
            D_model.eval()
            

            if batch_i % 4000 == 0:
                total = 0
                valid_loss_total , valid_real_total , valid_fake_total = 0 , 0 , 0
                for batch_j , valid_batch in enumerate(valid_data_loader):
                    total += 1
                    valid_loss , valid_real , valid_fake = train_one_batch( D_model , valid_batch , generator , opt ,perturb_std )
                    valid_loss_total += valid_loss.cpu().detach().numpy()
                    valid_real_total += valid_real.cpu().detach().numpy()
                    valid_fake_total += valid_fake.cpu().detach().numpy()
                    D_optimizer.zero_grad()

                print("Currently loss is " , valid_loss_total.item() / total ) 
                print("Currently real loss is " , valid_real_total.item() / total )
                print("Currently fake loss is " , valid_fake_total.item() / total )
                
                if  best_valid_loss > valid_loss_total.item() / total :
                    print("Loss Decreases so saving the file ...............----------->>>>>")        
                    state_dfs = D_model.state_dict()
                    torch.save(state_dfs,"Discriminator_checkpts/Attention_Disriminator_" + str(epoch) + ".pth.tar")
                    best_valid_loss = valid_loss_total.item() / total

######################################            
            
        
    
    
    
