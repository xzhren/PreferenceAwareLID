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

from config import *
from data_reader import *

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
import collections

tf.logging.set_verbosity(tf.logging.INFO)

def prepare_encoder_input(src_wids, src_sids, src_masks, params):
    src_vocab_size = params["src_vocab_size"]
    hidden_size = params["hidden_size"]
    number_of_classes = params["number_of_classes"]

    with tf.variable_scope('Source_Side'):
        src_emb = common_layers.embedding(src_wids, src_vocab_size, hidden_size)
    src_emb *= hidden_size**0.5
    
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

    domain_classifier_output = tf.reduce_sum(encoder_output*tf.expand_dims(shift_src_masks,2), 1) / tf.reduce_sum(shift_src_masks, 1, keep_dims=True) 
    domain_classifier_output = tf.layers.dense(domain_classifier_output, hidden_size, activation=tf.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    with tf.variable_scope('DomainClassifier', reuse=tf.AUTO_REUSE):
        tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
                hidden_size], initializer=\
                tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
        tag_vocab_bias = tf.get_variable("C_bias", shape=[number_of_classes], initializer=tf.zeros_initializer())
    logits = tf.nn.bias_add(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True), tag_vocab_bias)
    
    if use_user_lang_map == 1:
        logits = logits * src_sids
    elif use_user_lang_map == 2:
        model_logits = tf.nn.softmax(logits)
        domain_classifier_output = tf.concat([model_logits, src_sids], 1)

        with tf.variable_scope('UserDomainClassifier', reuse=tf.AUTO_REUSE):
            tag_vocab_embedding_tensor = tf.get_variable('C', [number_of_classes, \
                    number_of_classes*2], initializer=\
                    tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
            tag_vocab_bias = tf.get_variable("C_bias", shape=[number_of_classes], initializer=tf.zeros_initializer())

        logits = tf.nn.bias_add(tf.matmul(domain_classifier_output, tag_vocab_embedding_tensor, transpose_b=True), tag_vocab_bias)

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

    last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64) # shape: [batch_size, 1], values=0
    src_wids = tf.concat([features,last_padding],1) 
    src_sids = None
    if use_user_lang_map:
        src_sids = contexts
    src_masks = tf.to_float(tf.not_equal(src_wids,0))
    shift_src_masks = src_masks[:,:-1]
    shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

    logits, dist = output_layer(src_wids, src_sids, shift_src_masks, params)
    number_of_classes = params["number_of_classes"]
    targets = tf.one_hot(tf.cast(tf.squeeze(labels,1), tf.int32), depth=number_of_classes) 
   
    tf.logging.info("labels:"+str(labels))
    tf.logging.info("logits:"+str(logits))
    tf.logging.info("logits:"+str(tf.nn.softmax(logits)))
    tf.logging.info("targets:"+str(targets)) 
    
    xentropy_weighted = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = tf.reduce_sum(xentropy_weighted) / tf.cast(tf.shape(features)[0], dtype=tf.float32)
    res = tf.argmax(dist, tf.rank(dist) -1)
    accuracy = tf.metrics.accuracy(labels=labels,
                        predictions=res,
                        name='acc_op')

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

            if use_user_lang_map:
                tf.logging.info("feature shape:"+str(features))
                features, contexts = features['src'], features['scr']
                contexts = tf.cond(tf.shape(contexts)[0] < num_gpus, lambda: fill_until_num_gpus(contexts, num_gpus), lambda: contexts)
                context_shards = common_layers.approximate_split(contexts, num_gpus)
            features = tf.cond(tf.shape(features)[0] < num_gpus, lambda: fill_until_num_gpus(features, num_gpus), lambda: features)
            labels = tf.cond(tf.shape(labels)[0] < num_gpus, lambda: fill_until_num_gpus(labels, num_gpus), lambda: labels)
            feature_shards = common_layers.approximate_split(features, num_gpus)
            label_shards = common_layers.approximate_split(labels, num_gpus)
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
                        loss, train_acc = get_loss(feature_shards[i], context_shard, label_shards[i], params)
                        grads = optimizer.compute_gradients(loss)
                        #tf.get_variable_scope().reuse_variables()
                        loss_shards.append(loss)
                        grad_shards.append(grads)
                        train_acc_shards.append(train_acc)
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
            if use_user_lang_map:
                tf.logging.info("feature shape:"+str(features))
                features, contexts = features['src'], features['scr']
            last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
            src_wids = tf.concat([features,last_padding],1)
            src_sids = None
            if use_user_lang_map: src_sids = contexts
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
            tf.add_to_collection("labels", labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=tf.constant(0.0))

        if mode == tf.estimator.ModeKeys.PREDICT:
            features = features['feature']
            if use_user_lang_map:
                tf.logging.info("feature shape:"+str(features))
                features, contexts = features['src'], features['scr']
            last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
            src_wids = tf.concat([features,last_padding],1)
            src_sids = None
            if use_user_lang_map: src_sids = contexts
            src_masks = tf.to_float(tf.not_equal(src_wids,0))
            shift_src_masks = src_masks[:,:-1]
            shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

            _, prob = output_layer(src_wids, src_sids, shift_src_masks, params)
            res = tf.argmax(prob, tf.rank(prob) -1)

            output_label = tf.identity(res, name="output_label")
            predict_score = tf.identity(prob, name="predict_score")

            predictions = {'output_label': output_label, 'predict_score': predict_score}
            export_outputs = {'output': tf.estimator.export.PredictOutput(predictions)}

            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)




def add_dict_to_collection(collection_name, dict_):
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in dict_.items():
    tf.add_to_collection(key_collection, key)
    tf.add_to_collection(value_collection, value)

def GenerateSignature():
    features = tf.placeholder(tf.int64, [None, None], name="src_cid")
    last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
    src_cids = tf.concat([features,last_padding],1)

    src_wids, src_sids = None, None
    if use_user_lang_map:
        words = tf.placeholder(tf.int64, [None, None], name="src_wid")
        src_wids = tf.concat([words, last_padding],1)

    src_masks = tf.to_float(tf.not_equal(src_cids,0))
    shift_src_masks = src_masks[:,:-1]
    shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

    with tf.variable_scope('NmtModel') as var_scope:
        if use_word_embedding: _, prob = output_layer(src_cids, src_wids, shift_src_masks, params)
        else:  _, prob = output_layer(src_cids, [src_sids, src_wids], shift_src_masks, params)
    output_label = tf.argmax(prob, tf.rank(prob) -1)
    predict_score = prob

    output_label = tf.identity(output_label, name="output_label")
    predict_score = tf.identity(predict_score, name="predict_score")
    
    receiver_tensors = {
        "src_cid" : tf.saved_model.utils.build_tensor_info(features),
        "src_wid" : tf.saved_model.utils.build_tensor_info(words if use_user_lang_map else features),
    }
    export_outputs = {
        "output_label" : tf.saved_model.utils.build_tensor_info(output_label),
        "predict_score" : tf.saved_model.utils.build_tensor_info(predict_score)
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        receiver_tensors, export_outputs,
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    return {"langident_signature" : signature_def}
