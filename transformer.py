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
from data_reader import *
# from tools.script_target_matrix import script_target_matrix
import tensorflow as tf
import sys
import numpy as np
import common_attention
import common_layers
import subprocess
import re
from random import shuffle
from tensorflow.python.client import device_lib
from six.moves import xrange  # backward compatible with python2
#from dp import GraphDispatcher
import collections

tf.logging.set_verbosity(tf.logging.INFO)
#use_target_embedding = True
# use_target_embedding = False
#supported_lang=set(["ar","zh","zh-tw","nl","en","fr","de","he","hi","id","it","ja","ko","ms","pl","pt","ru","es","th","tr","ug","uk","vi"])
#supported_lang=None

def prepare_encoder_input(src_wids, src_sids, src_masks, params):
    src_vocab_size = params["src_vocab_size"]
    # script_vocab_size = params["script_vocab_size"]
    # src_word_vocab_size = params["src_word_vocab_size"]
    hidden_size = params["hidden_size"]
    number_of_classes = params["number_of_classes"]
    with tf.variable_scope('Source_Side'):
        src_emb = common_layers.embedding(src_wids, src_vocab_size, hidden_size)
        # if use_script_embedding: src_script_emb = common_layers.embedding(src_sids, script_vocab_size, hidden_size, 'ScriptEmbedding')
        # if use_word_embedding: src_word_emb = common_layers.embedding(src_sids, src_word_vocab_size, hidden_size, 'EnhancedWordEmbedding')
        # if use_word_script_embedding: 
        #     src_script_emb = common_layers.embedding(src_sids[0], script_vocab_size, hidden_size, 'ScriptEmbedding')
        #     src_word_emb = common_layers.embedding(src_sids[1], src_word_vocab_size, hidden_size, 'EnhancedWordEmbedding')
    src_emb *= hidden_size**0.5
    # if use_script_embedding:
    #     if use_target_embedding:
    #         with tf.variable_scope('ScriptDomainClassifier'):
    #             tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
    #                 hidden_size], initializer=\
    #                 tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
    #         script_target_matrix_tf = tf.constant(script_target_matrix, dtype=tf.float32) # script_size * number_of_classes
    #         script_target_embedding_weights = tf.matmul(script_target_matrix_tf, tag_vocab_embedding_tensor) # script_size * hidden_size
    #         script_target_embedding = tf.nn.embedding_lookup(script_target_embedding_weights, src_sids) # batch * timelen * hiddensize 
    #         script_target_embedding *= hidden_size**0.5

    #     src_script_emb *= hidden_size**0.5
    #     src_emb += src_script_emb
    #     if use_target_embedding: src_emb += script_target_embedding
    # if use_word_embedding:
    #     src_word_emb *= hidden_size**0.5
    #     # src_emb = tf.layers.dense(tf.concat([src_emb, src_word_emb], -1), hidden_size, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    #     # src_emb *= hidden_size**0.5
    #     # print("src_emb:", src_emb)
    #     # src_word_masks = tf.to_float(tf.not_equal(src_sids,1))
    #     # src_word_masks = tf.expand_dims(src_word_masks, 2)
    #     #src_word_emb = src_word_emb * src_word_masks + (1-src_word_masks) * src_emb
    #     # src_emb += src_word_emb * src_word_masks
    #     src_emb += src_word_emb
    # if use_word_script_embedding:
    #     # src_word_masks = tf.to_float(tf.not_equal(src_sids[1],1))
    #     # src_word_masks = tf.expand_dims(src_word_masks, 2)
    #     # src_word_emb = src_word_emb * src_word_masks
    #     # src_word_emb = src_word_emb * src_word_masks + (1-src_word_masks) * src_emb

    #     if use_target_embedding:
    #         with tf.variable_scope('ScriptDomainClassifier'):
    #             tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
    #                 hidden_size], initializer=\
    #                 tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
    #         script_target_matrix_tf = tf.constant(script_target_matrix, dtype=tf.float32) # script_size * number_of_classes
    #         script_target_embedding_weights = tf.matmul(script_target_matrix_tf, tag_vocab_embedding_tensor) # script_size * hidden_size
    #         script_target_embedding = tf.nn.embedding_lookup(script_target_embedding_weights, src_sids[0]) # batch * timelen * hiddensize
    #         script_target_embedding *= hidden_size**0.5

    #     src_script_emb *= hidden_size**0.5
    #     src_word_emb *= hidden_size**0.5
    #     src_emb += src_script_emb
    #     src_emb += src_word_emb
    #     if use_target_embedding: src_emb += script_target_embedding
    encoder_self_attention_bias = common_attention.attention_bias_ignore_padding(1-src_masks)
    encoder_input = common_attention.add_timing_signal_1d(src_emb)
    encoder_input = tf.multiply(encoder_input,tf.expand_dims(src_masks,2))
    return encoder_input,encoder_self_attention_bias

def layer_process(x, y, flag, dropout):
    if flag == None:
        return y
    for c in flag:
        if c == 'a':
            y = x+y
        elif c == 'n':
            y = common_layers.layer_norm(y)
        elif c == 'd':
            y = tf.nn.dropout(y, 1.0 - dropout)
    return y

def transformer_ffn_layer(x, params):
    filter_size = params["filter_size"]
    hidden_size = params["hidden_size"]
    relu_dropout = params["relu_dropout"]
    return common_layers.conv_hidden_relu(
            x,
            filter_size,
            hidden_size,
            dropout=relu_dropout)

def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        mask,
                        params={},
                        name="encoder"):
    num_hidden_layers = params["num_hidden_layers"]
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    prepost_dropout = params["prepost_dropout"]
    attention_dropout = params["attention_dropout"]
    preproc_actions = params['preproc_actions']
    postproc_actions = params['postproc_actions']
    x = encoder_input
    mask = tf.expand_dims(mask,2)
    with tf.variable_scope(name):
        for layer in xrange(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                o,w = common_attention.multihead_attention(
                        layer_process(None,x,preproc_actions,prepost_dropout),
                        None,
                        encoder_self_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout,
                        summaries=False,
                        name="encoder_self_attention")
                x = layer_process(x,o,postproc_actions,prepost_dropout)
                o = transformer_ffn_layer(layer_process(None,x,preproc_actions,prepost_dropout), params)
                #o,w = common_attention.multihead_attention(
                #        layer_process(None,x,preproc_actions,prepost_dropout),
                #        None,
                #        encoder_self_attention_bias,
                #        hidden_size,
                #        hidden_size,
                #        hidden_size,
                #        num_heads,
                #        attention_dropout,
                #        summaries=False,
                #        name="encoder_self_attention_ffn")
                x = layer_process(x,o,postproc_actions,prepost_dropout)
                x = tf.multiply(x,mask)
        return layer_process(None,x,preproc_actions,prepost_dropout)


def output_layer(src_wids, src_sids, shift_src_masks, params):
    encoder_input, encoder_self_attention_bias = prepare_encoder_input(src_wids, src_sids, shift_src_masks, params)
    encoder_input = tf.nn.dropout(encoder_input,\
                                        1.0 - params['prepost_dropout'])
    encoder_output = transformer_encoder(encoder_input, encoder_self_attention_bias,\
        shift_src_masks, params)
    hidden_size = params["hidden_size"]
    number_of_classes = params["number_of_classes"]
    #domain_classifier_output = tf.reduce_mean(encoder_output, 1)
    domain_classifier_output = tf.reduce_sum(encoder_output*tf.expand_dims(shift_src_masks,2), 1) / tf.reduce_sum(shift_src_masks, 1, keep_dims=True) 
    #domain_classifier_output = common_layers.domain_classifier(encoder_output, hidden_size)
    # if use_new_net and istanh:
    domain_classifier_output = tf.layers.dense(domain_classifier_output, hidden_size, activation=tf.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    if use_user_lang_map == 2:
        with tf.variable_scope('UserLangPerfer', reuse=tf.AUTO_REUSE):
            # if use_new_net and istanh:
                # user_logits = tf.layers.dense(src_sids, hidden_size, activation=tf.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            user_lang_perfer_tensor = tf.get_variable('C', [number_of_classes, \
                    hidden_size], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            user_lang_perfer_bias = tf.get_variable("C_bias", shape=[hidden_size], initializer=tf.zeros_initializer())
            user_logits = tf.tanh(tf.nn.bias_add(tf.matmul(src_sids, user_lang_perfer_tensor, transpose_b=False), user_lang_perfer_bias))
            # else:
            #     user_lang_perfer_tensor = tf.get_variable('C', [number_of_classes, \
            #             hidden_size], initializer=\
            #             tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            #     user_lang_perfer_bias = tf.get_variable("C_bias", shape=[hidden_size], initializer=tf.zeros_initializer())
            #     user_logits = tf.nn.bias_add(tf.matmul(src_sids, user_lang_perfer_tensor, transpose_b=False), user_lang_perfer_bias)
        domain_classifier_output = tf.concat([domain_classifier_output, user_logits], 1)

        with tf.variable_scope('DomainClassifier', reuse=tf.AUTO_REUSE):
            tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
                    hidden_size*2], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            # if use_new_net:
            tag_vocab_bias = tf.get_variable("C_bias", shape=[number_of_classes], initializer=tf.zeros_initializer())

        # if use_new_net:
        logits = tf.nn.bias_add(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True), tag_vocab_bias)
        # else:
            # logits = tf.nn.relu(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True))
    

    else:
        with tf.variable_scope('DomainClassifier', reuse=tf.AUTO_REUSE):
            tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
                    hidden_size], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            # if use_new_net:
            tag_vocab_bias = tf.get_variable("C_bias", shape=[number_of_classes], initializer=tf.zeros_initializer())

        # if use_new_net:
        logits = tf.nn.bias_add(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True), tag_vocab_bias)
        # else:
            # logits = tf.nn.relu(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True))
        
    if use_user_lang_map == 1:
        logits = logits * src_sids
    # elif use_user_lang_map == 10:
    #     logits = logits + src_sids
    elif use_user_lang_map == 3:
        with tf.variable_scope('UserLangPerfer', reuse=tf.AUTO_REUSE):
            # if use_new_net and istanh:
                # user_logits = tf.layers.dense(src_sids, hidden_size, activation=tf.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            user_lang_perfer_tensor = tf.get_variable('C', [number_of_classes, \
                    hidden_size], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            user_lang_perfer_bias = tf.get_variable("C_bias", shape=[hidden_size], initializer=tf.zeros_initializer())
            user_logits = tf.tanh(tf.nn.bias_add(tf.matmul(src_sids, user_lang_perfer_tensor, transpose_b=False), user_lang_perfer_bias))
            dist = tf.nn.softmax(logits)
            model_logits = tf.tanh(tf.nn.bias_add(tf.matmul(dist, user_lang_perfer_tensor, transpose_b=False), user_lang_perfer_bias))
            # else:
            #     user_lang_perfer_tensor = tf.get_variable('C', [number_of_classes, \
            #             hidden_size], initializer=\
            #             tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            #     user_lang_perfer_bias = tf.get_variable("C_bias", shape=[hidden_size], initializer=tf.zeros_initializer())
            #     user_logits = tf.nn.bias_add(tf.matmul(src_sids, user_lang_perfer_tensor, transpose_b=False), user_lang_perfer_bias)
            #     dist = tf.nn.softmax(logits)
            #     model_logits = tf.nn.bias_add(tf.matmul(dist, user_lang_perfer_tensor, transpose_b=False), user_lang_perfer_bias)
        domain_classifier_output = tf.concat([model_logits, user_logits], 1)

        with tf.variable_scope('UserDomainClassifier', reuse=tf.AUTO_REUSE):
            tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
                    hidden_size*2], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            # if use_new_net:
            tag_vocab_bias = tf.get_variable("C_bias", shape=[number_of_classes], initializer=tf.zeros_initializer())

        # if use_new_net:
        logits = tf.nn.bias_add(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True), tag_vocab_bias)
        # else:
            # logits = tf.nn.relu(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True))

    elif use_user_lang_map == 4:
        model_logits = tf.nn.softmax(logits)
        domain_classifier_output = tf.concat([model_logits, src_sids], 1)

        with tf.variable_scope('UserDomainClassifier', reuse=tf.AUTO_REUSE):
            tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
                    number_of_classes*2], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            # if use_new_net:
            tag_vocab_bias = tf.get_variable("C_bias", shape=[number_of_classes], initializer=tf.zeros_initializer())

        # if use_new_net:
        logits = tf.nn.bias_add(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True), tag_vocab_bias)
        # else:
            # logits = tf.nn.relu(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True))



    dist = tf.nn.softmax(logits)
    dist = tf.clip_by_value(dist, 1e-8, 1.0-1e-8)
    return logits, dist


def compute_batch_indices(batch_size, beam_size):
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_sum(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def get_loss(features, contexts, labels, params):
    trg_vocab_file = params['vocab_trg']
    trg_vocab = open(trg_vocab_file).readlines()
    trg_vocab = [v.strip() for v in trg_vocab]
    tf.logging.info("trg_vocab:"+str(trg_vocab))
    tf.logging.info("supported_lang:"+str(supported_lang))
    if supported_lang: weighted_mask = [[1.0 if v in supported_lang else 0.5 for v in trg_vocab]]
    else: weighted_mask = [[1.0]*len(trg_vocab)]
    tf.logging.info("weighted_mask:"+str(weighted_mask))

    #features = tf.Print(features,[tf.shape(features)])
    last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64) # shape: [batch_size, 1], values=0
    src_wids = tf.concat([features,last_padding],1) 
    src_sids = None
    # if use_script_embedding or use_word_embedding:
    #     src_sids = tf.concat([contexts,last_padding],1) 
    # if use_word_script_embedding:
    #     src_sids = [tf.concat([contexts[0],last_padding],1), tf.concat([contexts[1],last_padding],1)]
    if use_user_lang_map:
        src_sids = contexts
    src_masks = tf.to_float(tf.not_equal(src_wids,0))
    shift_src_masks = src_masks[:,:-1]
    shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)
    #trg_wids = tf.concat([labels,last_padding],1)
    #trg_masks = tf.to_float(tf.not_equal(trg_wids,0))
    #shift_trg_masks = trg_masks[:,:-1]
    #shift_trg_masks = tf.pad(shift_trg_masks,[[0,0],[1,0]],constant_values=1)


    logits, dist = output_layer(src_wids, src_sids, shift_src_masks, params)
    number_of_classes = params["number_of_classes"]
    targets = tf.one_hot(tf.cast(tf.squeeze(labels,1), tf.int32), depth=number_of_classes) 
   
    tf.logging.info("labels:"+str(labels))
    tf.logging.info("logits:"+str(logits))
    tf.logging.info("logits:"+str(tf.nn.softmax(logits)))
    tf.logging.info("targets:"+str(targets)) 
    #weighted_xentropy = tf.nn.weighted_cross_entropy_with_logits(logits=tf.nn.softmax(tf.expand_dims(logits,1)), targets=targets, pos_weight=weighted_mask)
    #loss_weighted = tf.reduce_sum(weighted_xentropy) / tf.cast(tf.shape(features)[0], dtype=tf.float32)
    
    #xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    
    if supported_lang:
        #weighted_mask = tf.tile(weighted_mask, [tf.shape(features)[0],1])
        targets = tf.cast(tf.squeeze(labels,1), tf.int32)
        weights = tf.gather(weighted_mask[0], targets)
        xentropy_weighted = tf.losses.sparse_softmax_cross_entropy(targets, logits, weights) 
    #xentropy_weighted = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets*weighted_mask)
    else:
        xentropy_weighted = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    #xentropy_weighted = targets * -tf.log(dist) * weighted_mask #+ (1-targets) * -tf.log(1-dist)
    #xentropy_weighted = targets * (1-dist) * -tf.log(dist) * weighted_mask #+ (1-targets) * (dist) * -tf.log(1-dist)
    #tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=targets, pos_weight=weighted_mask)
    
    #xentropy_limit = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets*mask_lang)
    #loss_raw = tf.reduce_sum(xentropy) / tf.cast(tf.shape(features)[0], dtype=tf.float32)
    loss_weighted = tf.reduce_sum(xentropy_weighted) / tf.cast(tf.shape(features)[0], dtype=tf.float32)
    #loss_limit = tf.reduce_sum(xentropy_limit) / tf.cast(tf.shape(features)[0], dtype=tf.float32)
    #loss = (loss_raw + loss_limit) / 2

    loss = loss_weighted
    #loss = loss_raw
    #dist = tf.nn.softmax(logits)
    res = tf.argmax(dist, tf.rank(dist) -1)
    accuracy = tf.metrics.accuracy(labels=labels,
                        predictions=res,
                        name='acc_op')
    #train_accuary = tf.cast(tf.count_nonzero(res - labels), dtype = tf.float32) / tf.cast(tf.size(labels, out_type = tf.int32 ), dtype=tf.float32)
    #return  batch_cost_tag, error

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy[1])
    return loss, accuracy


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def transformer_model_fn(features, labels, mode, params):
    with tf.variable_scope('NmtModel') as var_scope:

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_gpus = params['num_gpus']
            gradient_clip_value = params['gradient_clip_value']
            step = tf.to_float(tf.train.get_global_step())
            warmup_steps = params['warmup_steps']
            if params['learning_rate_decay'] == 'sqrt':
                lr_warmup = params['learning_rate_peak'] * tf.minimum(1.0,step/warmup_steps)
                lr_decay = params['learning_rate_peak'] * tf.minimum(1.0,tf.sqrt(warmup_steps/step))
                lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
            elif params['learning_rate_decay'] == 'exp':
                lr = tf.train.exponential_decay(params['learning_rate_peak'],
                        global_step=step,
                        decay_steps=params['decay_steps'],
                        decay_rate=params['decay_rate'])
            else:
                tf.logging.info("learning rate decay strategy not supported")
                sys.exit()
            if params['optimizer'] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif params['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.997, epsilon=1e-09)
            else:
                tf.logging.info("optimizer not supported")
                sys.exit()
            #optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-09)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, gradient_clip_value)

            def fill_until_num_gpus(inputs, num_gpus):
                outputs = inputs
                for i in range(num_gpus - 1):
                    outputs = tf.concat([outputs, inputs], 0)
                outputs= outputs[:num_gpus,]
                return outputs

            # if use_script_embedding or use_word_embedding or use_user_lang_map:
            if use_user_lang_map:
                tf.logging.info("feature shape:"+str(features))
                features, contexts = features['src'], features['scr']
                contexts = tf.cond(tf.shape(contexts)[0] < num_gpus, lambda: fill_until_num_gpus(contexts, num_gpus), lambda: contexts)
                context_shards = common_layers.approximate_split(contexts, num_gpus)
            # if use_word_script_embedding:
            #     tf.logging.info("feature shape:"+str(features))
            #     features, contexts1, contexts2 = features['src'], features['scr1'], features['scr2']
            #     contexts1 = tf.cond(tf.shape(contexts1)[0] < num_gpus, lambda: fill_until_num_gpus(contexts1, num_gpus), lambda: contexts1) 
            #     contexts2 = tf.cond(tf.shape(contexts2)[0] < num_gpus, lambda: fill_until_num_gpus(contexts2, num_gpus), lambda: contexts2) 
            #     context_shards = [common_layers.approximate_split(contexts1, num_gpus), common_layers.approximate_split(contexts2, num_gpus)]
            features = tf.cond(tf.shape(features)[0] < num_gpus, lambda: fill_until_num_gpus(features, num_gpus), lambda: features)
            labels = tf.cond(tf.shape(labels)[0] < num_gpus, lambda: fill_until_num_gpus(labels, num_gpus), lambda: labels)
            feature_shards = common_layers.approximate_split(features, num_gpus)
            label_shards = common_layers.approximate_split(labels, num_gpus)
            #loss_shards = dispatcher(get_loss, feature_shards, label_shards, params)
            devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
            loss_shards = []
            grad_shards = []
            train_acc_shards = []
            for i, device in enumerate(devices):
                #if i > 0:
                    #var_scope.reuse_variables()
                with tf.variable_scope( tf.get_variable_scope(), reuse=True if i > 0 else None):
                    with tf.device(device):
                        context_shard = None
                        if use_user_lang_map: context_shard = context_shards[i]
                        # if use_script_embedding or use_word_embedding or use_user_lang_map: context_shard = context_shards[i]
                        # if use_word_script_embedding: context_shard = [context_shards[0][i], context_shards[1][i]]
                        loss, train_acc = get_loss(feature_shards[i], context_shard, label_shards[i], params)
                        grads = optimizer.compute_gradients(loss)
                        #tf.get_variable_scope().reuse_variables()
                        loss_shards.append(loss)
                        grad_shards.append(grads)
                        train_acc_shards.append(train_acc)
            #loss_shards = tf.Print(loss_shards,[loss_shards])
            loss = tf.reduce_mean(loss_shards)
            grad = average_gradients(grad_shards)
            train_acc = tf.reduce_mean(train_acc_shards)
            train_op = optimizer.apply_gradients(grad, global_step=tf.train.get_global_step())
            if params['ema_decay'] > 0.0:
                ema = tf.train.ExponentialMovingAverage(decay=params['ema_decay'], num_updates=tf.train.get_global_step())
                with tf.control_dependencies([train_op]):
                    train_op = ema.apply(tf.trainable_variables())
            logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : train_acc}, every_n_iter=100)
            summary_hook = tf.train.SummarySaverHook(save_steps=100, summary_op=tf.summary.merge_all()) 
            


            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            if init_checkpoint:
                (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)



            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks = [logging_hook, summary_hook])

        if mode == tf.estimator.ModeKeys.EVAL:
            # if use_script_embedding or use_word_embedding or use_user_lang_map:
            if use_user_lang_map:
                tf.logging.info("feature shape:"+str(features))
                features, contexts = features['src'], features['scr']
            # if use_word_script_embedding:
            #     tf.logging.info("feature shape:"+str(features))
            #     features, contexts1, contexts2 = features['src'], features['scr1'], features['scr2']
            last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
            src_wids = tf.concat([features,last_padding],1)
            src_sids = None
            # if use_script_embedding or use_word_embedding:
            #     src_sids = tf.concat([contexts,last_padding],1)
            # if use_word_script_embedding:
            #     src_sids = [tf.concat([contexts1,last_padding],1), tf.concat([contexts2,last_padding],1)]
            if use_user_lang_map:
                src_sids = contexts
            src_masks = tf.to_float(tf.not_equal(src_wids,0))
            shift_src_masks = src_masks[:,:-1]
            shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

            _, prob = output_layer(src_wids, src_sids, shift_src_masks, params)
            res = tf.argmax(prob, tf.rank(prob) -1)
            accuracy = tf.metrics.accuracy(labels=labels,
                              predictions=res,
                              name='acc_op')
            predictions = {"predict_score": prob }
            predict_label = {"label": res }
            predict_accuracy = {"accuracy": accuracy[1] }
            add_dict_to_collection("predictions", predictions)
            add_dict_to_collection("predict_label", predict_label)
            add_dict_to_collection("accuracy", predict_accuracy)
            tf.add_to_collection("features", features)
            if use_user_lang_map: tf.add_to_collection("contexts", contexts)
            # if use_script_embedding or use_word_embedding or use_user_lang_map: tf.add_to_collection("contexts", contexts)
            # if use_word_script_embedding: tf.add_to_collection("contexts1", contexts1); tf.add_to_collection("contexts2", contexts2)
            tf.add_to_collection("labels", labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=tf.constant(0.0))

        if mode == tf.estimator.ModeKeys.PREDICT:
            features = features['feature']
            # if use_script_embedding or use_word_embedding or use_user_lang_map:
            if use_user_lang_map:
                tf.logging.info("feature shape:"+str(features))
                features, contexts = features['src'], features['scr']
            # if use_word_script_embedding:
            #     tf.logging.info("feature shape:"+str(features))
            #     features, contexts1, contexts2 = features['src'], features['scr1'], features['scr2']
            last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
            src_wids = tf.concat([features,last_padding],1)
            src_sids = None
            # if use_script_embedding or use_word_embedding:
            #     src_sids = tf.concat([contexts,last_padding],1)
            # if use_word_script_embedding:
            #     src_sids = [tf.concat([contexts1,last_padding],1), tf.concat([contexts2,last_padding],1)]
            if use_user_lang_map:
                src_sids = contexts
            src_masks = tf.to_float(tf.not_equal(src_wids,0))
            shift_src_masks = src_masks[:,:-1]
            shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

            _, prob = output_layer(src_wids, src_sids, shift_src_masks, params)
            res = tf.argmax(prob, tf.rank(prob) -1)

            output_label = tf.identity(res, name="output_label")
            predict_score = tf.identity(prob, name="predict_score")

            predictions = {'output_label': output_label, 'predict_score': predict_score}
            #export_outputs={'output_label': tf.estimator.export.PredictOutput(predictions['output_label']), 
            #               'predict_score': tf.estimator.export.PredictOutput(predictions['predict_score'])}
            export_outputs = {'output': tf.estimator.export.PredictOutput(predictions)}

            #for op in tf.get_default_graph().get_operations():
            #    print(str(op.name))

            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)




def add_dict_to_collection(collection_name, dict_):
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in dict_.items():
    tf.add_to_collection(key_collection, key)
    tf.add_to_collection(value_collection, value)

def GenerateSignature():
    #src_tensor_test_ph = tf.placeholder(tf.int64, [None, None], name="src_cid")
    ##src_mask_test_ph = tf.to_float(tf.placeholder(tf.int64, [None, None], name="src_mask"))

    #features = src_tensor_test_ph
    #last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
    #src_wids = tf.concat([features,last_padding],1)
    
    src_cids = None
    if True:
        features = tf.placeholder(tf.int64, [None, None], name="src_cid")
        last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
        src_cids = tf.concat([features,last_padding],1)

    #src_sids = None
    #if use_script_embedding or use_word_embedding:
    #    contexts = tf.placeholder(tf.int64, [None, None], name="src_sid")
    #    src_sids = tf.concat([contexts,last_padding],1)

    src_wids, src_sids = None, None
    if use_user_lang_map:
    # if use_word_script_embedding or use_word_embedding:
        words = tf.placeholder(tf.int64, [None, None], name="src_wid")
        src_wids = tf.concat([words, last_padding],1)
    # if use_word_script_embedding or use_script_embedding or use_user_lang_map:
        # scripts = tf.placeholder(tf.int64, [None, None], name="src_sid")
        # src_sids = tf.concat([scripts,last_padding],1)


    src_masks = tf.to_float(tf.not_equal(src_cids,0))
    shift_src_masks = src_masks[:,:-1]
    shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

    with tf.variable_scope('NmtModel') as var_scope:
        if use_word_embedding: _, prob = output_layer(src_cids, src_wids, shift_src_masks, params)
        # elif use_script_embedding: _, prob = output_layer(src_cids, src_sids, shift_src_masks, params)
        # elif use_word_script_embedding: _, prob = output_layer(src_cids, [src_sids, src_wids], shift_src_masks, params)
        else:  _, prob = output_layer(src_cids, [src_sids, src_wids], shift_src_masks, params)
    output_label = tf.argmax(prob, tf.rank(prob) -1)
    predict_score = prob

    #src_tensor_test_ph = tf.identity(src_tensor_test_ph, name="src_wid")
    #src_mask_test_ph = tf.identity(src_mask_test_ph, name="src_mask")
    output_label = tf.identity(output_label, name="output_label")
    predict_score = tf.identity(predict_score, name="predict_score")
    
    receiver_tensors = {
        "src_cid" : tf.saved_model.utils.build_tensor_info(features),
        "src_wid" : tf.saved_model.utils.build_tensor_info(words if use_user_lang_map else features),
        # "src_wid" : tf.saved_model.utils.build_tensor_info(words if use_word_script_embedding or use_word_embedding else features),
        # "src_sid" : tf.saved_model.utils.build_tensor_info(scripts if use_word_script_embedding or use_script_embedding else features),
    #    "src_mask" : tf.saved_model.utils.build_tensor_info(src_mask_test_ph)
    #    "src_sid" : tf.saved_model.utils.build_tensor_info(contexts if use_script_embedding or use_word_embedding else src_tensor_test_ph),
    }
    export_outputs = {
        "output_label" : tf.saved_model.utils.build_tensor_info(output_label),
        "predict_score" : tf.saved_model.utils.build_tensor_info(predict_score)
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        receiver_tensors, export_outputs,
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    return {"langident_signature" : signature_def}

if __name__ == '__main__':
    trg_vocab_file = "corpus104_23/label.txt"
    trg_vocab = open(trg_vocab_file).readlines()
    trg_vocab = [v.strip() for v in trg_vocab]
    print(trg_vocab)
    print(supported_lang)
    mask_lang = [[1.0 if v in supported_lang else 0.0 for v in trg_vocab]]
    print(mask_lang)
    mask_lang = np.array(mask_lang)
    print(mask_lang.shape)
