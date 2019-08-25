#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:28:33 2019

@author: r17935avinash
"""

import torch
import logging
from pykp.io import KeyphraseDataset
from torch.utils.data import DataLoader

vocab_file = "data/kp20k_separated"
vocab_size=50002
data_file = "data/kp20k_separated'
data_filename_suffix = False
batch_size = 16
custom_vocab_filename_suffix = False

opt = Namespace(attn_mode='concat', baseline='self', batch_size=32, batch_workers=4, bidirectional=True, bridge='copy', checkpoint_interval=4000, copy_attention=True, copy_input_feeding=False, coverage_attn=False, coverage_loss=False, custom_data_filename_suffix=False, custom_vocab_filename_suffix=False, data='data/kp20k_separated/', data_filename_suffix='', dec_layers=1, decay_method='', decoder_size=300, decoder_type='rnn', delimiter_type=0, delimiter_word='<sep>', device=device(type='cuda', index=0), disable_early_stop_rl=False, dropout=0.1, dynamic_dict=True, early_stop_tolerance=4, enc_layers=1, encoder_size=150, encoder_type='rnn', epochs=20, exp='kp20k.rl.one2many.cat.copy.bi-directional', exp_path='exp/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', final_perturb_std=0, fix_word_vecs_dec=False, fix_word_vecs_enc=False, goal_vector_mode=0, goal_vector_size=16, gpuid=0, init_perturb_std=0, input_feeding=False, lambda_coverage=1, lambda_orthogonal=0.03, lambda_target_encoder=0.03, learning_rate=0.001, learning_rate_decay=0.5, learning_rate_decay_rl=False, learning_rate_rl=5e-05, loss_normalization='tokens', manager_mode=1, match_type='exact', max_grad_norm=1, max_length=60, max_sample_length=6, max_unk_words=1000, mc_rollouts=False, model_path='model/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', must_teacher_forcing=False, num_predictions=1, num_rollouts=3, one2many=True, one2many_mode=1, optim='adam', orthogonal_loss=False, param_init=0.1, perturb_baseline=False, perturb_decay_factor=0.0001, perturb_decay_mode=1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, pretrained_model='model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model', regularization_factor=0.0, regularization_type=0, remove_src_eos=False, replace_unk=True, report_every=10, review_attn=False, reward_shaping=False, reward_type=7, save_model='model', scheduled_sampling=False, scheduled_sampling_batches=10000, seed=9527, separate_present_absent=True, share_embeddings=True, source_representation_queue_size=128, source_representation_sample_size=32, start_checkpoint_at=2, start_decay_at=8, start_epoch=1, target_encoder_size=64, teacher_forcing_ratio=0, timemark='20190701-192604', title_guided=False, topk='G', train_from='', train_ml=False, train_rl=True, truncated_decoder=0, use_target_encoder=False, vocab='data/kp20k_separated/', vocab_filename_suffix='', vocab_size=50002, warmup_steps=4000, word_vec_size=100, words_min_frequency=0)

def load_vocab():
    # load vocab
    logging.info("Loading vocab from disk: %s" % (vocab_file))
    if not custom_vocab_filename_suffix:
        word2idx, idx2word, vocab = torch.load(vocab_file + '/vocab.pt', 'wb')
    else:
        word2idx, idx2word, vocab = torch.load(vocab_file + '/vocab.%s.pt' % opt.vocab_filename_suffix, 'wb')
    # assign vocab to opt
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % vocab_size)

    return word2idx, idx2word, vocab

def load_data_and_vocab(opt, load_train=True):
    # load vocab
    #print(opt)
    word2idx, idx2word, vocab = load_vocab()

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % data_file)

    if load_train:  # load training dataset
        if not opt.one2many:  # load one2one dataset
            if not opt.custom_data_filename_suffix:
                train_one2one = torch.load(data_file+ '/train.one2one.pt', 'wb')
            else:
                train_one2one = torch.load(data_file+ '/train.one2one.%s.pt' % data_filename_suffix, 'wb')
            train_one2one_dataset = KeyphraseDataset(train_one2one, word2idx=word2idx, idx2word=idx2word, type='one2one', load_train=load_train, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
            train_loader = DataLoader(dataset=train_one2one_dataset,
                                              collate_fn=train_one2one_dataset.collate_fn_one2one,
                                              num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                              shuffle=True)
            logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

            if not opt.custom_data_filename_suffix:
                valid_one2one = torch.load(data_file + '/valid.one2one.pt', 'wb')
            else:
                valid_one2one = torch.load(data_file + '/valid.one2one.%s.pt' % data_filename_suffix, 'wb')
            valid_one2one_dataset = KeyphraseDataset(valid_one2one, word2idx=word2idx, idx2word=idx2word,
                                                     type='one2one', load_train=load_train, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
            valid_loader = DataLoader(dataset=valid_one2one_dataset,
                                      collate_fn=valid_one2one_dataset.collate_fn_one2one,
                                      num_workers=opt.batch_workers, batch_size=batch_size, pin_memory=True,
                                      shuffle=False)
            logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))


        else:  # load one2many dataset
            if not opt.custom_data_filename_suffix:
                train_one2many = torch.load(data_file + '/train.one2many.pt', 'wb')
            else:
                train_one2many = torch.load(data_file + '/train.one2many.%s.pt' % data_filename_suffix, 'wb')
            train_one2many_dataset = KeyphraseDataset(train_one2many, word2idx=word2idx, idx2word=idx2word, type='one2many', delimiter_type=opt.delimiter_type, load_train=load_train, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
            train_loader = DataLoader(dataset=train_one2many_dataset,
                                      collate_fn=train_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=batch_size, pin_memory=True,
                                      shuffle=True)
            logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

            if not opt.custom_data_filename_suffix:
                valid_one2many = torch.load(data_file + '/valid.one2many.pt', 'wb')
            else:
                valid_one2many = torch.load(data_file + '/valid.one2many.%s.pt' % data_filename_suffix, 'wb')
            #valid_one2many = valid_one2many[:2000]
            valid_one2many_dataset = KeyphraseDataset(valid_one2many, word2idx=word2idx, idx2word=idx2word,
                                                      type='one2many', delimiter_type=opt.delimiter_type, load_train=load_train, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
            valid_loader = DataLoader(dataset=valid_one2many_dataset,
                                      collate_fn=valid_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=batch_size, pin_memory=True,
                                      shuffle=False)
            logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, valid_loader, word2idx, idx2word, vocab
    else:
        if not opt.custom_data_filename_suffix:
            test_one2many = torch.load(data_file+ '/test.one2many.pt', 'wb')
        else:
            test_one2many = torch.load(data_file + '/test.one2many.%s.pt' % data_filename_suffix, 'wb')
        test_one2many_dataset = KeyphraseDataset(test_one2many, word2idx=word2idx, idx2word=idx2word,
                                                      type='one2many', delimiter_type=opt.delimiter_type, load_train=load_train, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
        test_loader = DataLoader(dataset=test_one2many_dataset,
                                      collate_fn=test_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=batch_size, pin_memory=True,
                                      shuffle=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

        return test_loader, word2idx, idx2word, vocab