#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
#
# History:
# 2019.07.22. Be created by xingzhang.rxz. Used for language identification in CNN method.
# 2018.04.27. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor.
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import *
from transformer import *
import tensorflow as tf
from random import shuffle
import random
import os
from six.moves import xrange  # backward compatible with python2

tf.logging.set_verbosity(tf.logging.INFO)

def shuffle_train(train_src, train_trg, suffix, seed=8888):
    cmd = 'sh tools/shuffle-train.sh %s %s %s %s' % (train_src, train_trg, suffix, seed)
    p = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.communicate()

def main(_):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session_config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True, log_device_placement=False)
    trainEstimator = tf.estimator.Estimator(model_fn=transformer_model_fn,\
            config=tf.estimator.RunConfig(\
            save_checkpoints_steps=params['save_checkpoints_steps'],\
            keep_checkpoint_max = params['keep_checkpoint_max'],session_config=session_config),\
            model_dir=model_dir,params=params)

    epoch = 0
    while True:
        epoch += 1
        if epoch >= params['epoch']:
            break
        tf.logging.info("Epoch %i", epoch)
        if params['shuffle_train']:
            tf.logging.info("Shuffling data for Epoch %i", epoch)
            random.seed(params['seed'])
            states=random.sample(xrange(100000), params['epoch'])
            if not os.path.isfile(train_src+'.epoch'+str(epoch)) or not os.path.isfile(train_trg+'.epoch'+str(epoch)):
                shuffle_train(train_src,train_trg,'epoch'+str(epoch),states[epoch])
        train_input_fn = lambda: input_fn(
            train_src+'.epoch'+str(epoch) if params['shuffle_train'] else train_src,
            train_trg+'.epoch'+str(epoch) if params['shuffle_train'] else train_trg,
            vocab_src,
            vocab_trg,
            ulid_file=train_ulid,
            batch_size_words=params['train_batch_size_words'],
            max_len=params['train_max_len'],
            num_gpus=params['num_gpus'],
            is_train=True,
            use_script_embedding=None,
            use_word_embedding=None,
            script_vocab_file=None,
            src_word_vocab_file=None
        )
        trainEstimator.train(train_input_fn)
        if not params['keep_track_dataflow']:
            cmd = 'rm -f %s %s' % (train_src+'.epoch'+str(epoch), train_trg+'.epoch'+str(epoch))
            p = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            p.communicate()

if __name__ == '__main__':
    tf.app.run()

