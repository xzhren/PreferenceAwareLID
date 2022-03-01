#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2022 Alibaba Inc. All Rights Reserved.
#
# History:
# 2022.03.01. Be created by xingzhang.rxz. Used for language identification.
# 2018.04.27. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor.
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="exp1")
parser.add_argument('--corpus_dir', type=str, default="corpus")
parser.add_argument('--vocab_size', type=int, default=409)
parser.add_argument('--class_num', type=int, default=11)
parser.add_argument('--postfix', type=str, default="")
parser.add_argument('--vocab_dir', type=str, default="")
parser.add_argument('--train_batch', type=int, default=102400)
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--ema_decay', type=float, default=0.9999)
parser.add_argument('--warmup_steps', type=int, default=8000)
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--init_checkpoint', type=str, default=None)
parser.add_argument('--init_step', type=int, default=0)
parser.add_argument('--num_hidden_layers', type=int, default=1)
parser.add_argument('--use_multiscale_att', type=bool, default=False)
parser.add_argument('--use_user_lang_map', type=int, default=0)
parser.add_argument('--showinfos', type=bool, default=False)
parser.add_argument('--prefetch', type=int, default=1000000)
parser.add_argument('--export', type=bool, default=False)
parser.add_argument('--exportstep', type=str, default=None)
parser.add_argument('--export_dir', type=str, default=None)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--filter_size', type=int, default=2048)

args = parser.parse_args()

if args.vocab_dir == "":
    args.vocab_dir = args.corpus_dir

basedir = args.corpus_dir
train_src = basedir+'/train.src'
train_trg = basedir+'/train.trg'
train_ulid = basedir+'/train.ulid'
dev_src = basedir+'/test.src'
dev_trg = basedir+'/test.trg'
dev_ulid = basedir+'/test.ulid'
vocab_src = args.vocab_dir+'/vocab.txt'
vocab_trg = args.vocab_dir+'/label.txt'
model_dir = './logs/'+args.exp_name
dev_out = model_dir+'/eval/eval'+args.postfix+'.out'
acc_log = model_dir+'/acc'+args.postfix+'.log'
init_checkpoint = args.init_checkpoint
use_user_lang_map = args.use_user_lang_map
use_multiscale_att = args.use_multiscale_att
showinfos = args.showinfos
export = args.export
exportstep = args.exportstep
start_examples = args.init_step * args.train_batch / args.max_len

params = {}
params['vocab_trg'] = vocab_trg
params["num_gpus"] = 1
params["epoch"] = 100
params["save_checkpoints_steps"]=5000
params["keep_checkpoint_max"]=30
params['train_max_len'] = args.max_len
params["train_batch_size_words"]=args.train_batch
params["optimizer"] = 'adam'            # adam or sgd
params["learning_rate_decay"] = 'sqrt'  # sqrt: 0->peak->sqrt; exp: peak->exp, used for finetune
params["learning_rate_peak"] = args.lr
params["warmup_steps"] = args.warmup_steps           # only for sqrt decay
params["decay_steps"] = 100             # only for exp decay, decay every n steps
params["decay_rate"] = 0.9              # only for exp decay
params["src_vocab_size"] = args.vocab_size                #Must same with line numbers vocab.txt 
params["hidden_size"] = args.hidden_size
params["filter_size"] = args.filter_size
params["num_hidden_layers"] = args.num_hidden_layers
params["num_heads"] = args.num_heads
params['gradient_clip_value'] = 5.0
params["confidence"] = 0.9              # label smoothing confidence
params["prepost_dropout"] = 0.1 if not export else 0.0
params["relu_dropout"] = 0.1 if not export else 0.0
params["attention_dropout"] = 0.1 if not export else 0.0
params["preproc_actions"] = 'n'         # layer normalization
params["postproc_actions"] = 'da'       # dropout; residual connection
params["number_of_classes"]=args.class_num      # Must same with line numbers label.txt

params["shuffle_train"] = args.shuffle
params['keep_track_dataflow'] = True
params["continuous_eval"] = True        # True for training
params["eval_from_step"] = 0            # If > 0, evaluation will start at step (included) indicated by this parameter
params['ema_decay'] = args.ema_decay            # If > 0.0, use exponential moving average for training, set to 0.0 to disable
params["beam_size"] = 6
params["alpha"] = 0.6
params["decode_batch_size"] = 10
params["max_decoded_trg_len"] = 100
params['seed'] = 8888
