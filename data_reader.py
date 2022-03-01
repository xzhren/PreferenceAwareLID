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

import sys
import codecs
import tensorflow as tf
import random
from config import *


def _replace_str(str_raw):
    return str_raw.replace("\xa0", "").replace("\u2028", "").replace("\x85", "").replace("\u2009", "").replace("\u202f", "")

def paste_text_file_to_chars(data_src, is_train):
  for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
    if is_train and lineid < start_examples: continue
    tmpline = line.lower().rstrip("\n").replace(" ", "")
    tmpline = _replace_str(tmpline)
    yield list(tmpline)

def paste_text_file_to_labels(data_src, is_train):
  for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
    if is_train and lineid < start_examples: continue
    yield [line.rstrip("\n")]

def paste_text_file_to_ulids(data_src, is_train):
  langlt = [lang.strip() for lang in open(vocab_trg).readlines()]
  print(langlt)
  for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
      ulang_data = {}
      if is_train:
          for item in line.strip().split(","):
            infos = item.split(":") 
            if random.random() > 0.5:
                ulang_data[infos[0]] = float(infos[1])
            else:
                ulang_data[infos[0]] = 1/len(langlt)
      else:
          total = 0.0
          sep_str = ","
          line = line.strip()
          if sep_str not in line:
            sep_str = ";"
            line = line.strip(";")
          for item in line.split(sep_str):
            infos = item.split(":") 
            if len(infos) != 2: print(line, item)
            total += float(infos[1])
            ulang_data[infos[0]] = float(infos[1])
          for k,v in ulang_data.items():
            ulang_data[k] = v/total
      langres = [0.0] * len(langlt)
      for index, lang in enumerate(langlt):
        langres[index] = ulang_data[lang]
      yield langres

def input_fn(src_file,
             trg_file,
             src_vocab_file,
             trg_vocab_file,
             ulid_file=None,
             num_buckets=20,
             max_len=100,
             batch_size=200,
             batch_size_words=4096,
             num_gpus=1,
             is_train=True,
             vocab_granularity='char',
             use_script_embedding=False,
             use_word_embedding=False,
             script_vocab_file=None,
             src_word_vocab_file=None):
    src_vocab = tf.contrib.lookup.index_table_from_file(src_vocab_file,default_value=1) # NOTE 'eos' -> 0,  unk->1
    trg_vocab = tf.contrib.lookup.index_table_from_file(trg_vocab_file,default_value=1)
    
    src_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_chars(data_src=src_file, is_train=is_train), tf.string, tf.TensorShape([None]))
    trg_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_labels(data_src=trg_file, is_train=is_train), tf.string, tf.TensorShape([None]))

    if use_user_lang_map:
      ulid_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_ulids(data_src=ulid_file, is_train=is_train), tf.float32, tf.TensorShape([None]))
      src_trg_dataset = tf.data.Dataset.zip((src_dataset, ulid_dataset, trg_dataset))       
    else:
      src_trg_dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
      
    if use_user_lang_map:
      src_trg_dataset = src_trg_dataset.map(
            lambda src, srcs, trg: (src_vocab.lookup(src), srcs, trg_vocab.lookup(trg)),
            num_parallel_calls=10).prefetch(args.prefetch)
    else:
      src_trg_dataset = src_trg_dataset.map(
            lambda src, trg: (src_vocab.lookup(src), trg_vocab.lookup(trg)),
            num_parallel_calls=10).prefetch(args.prefetch)
    
    if is_train == True:
      def key_func(src_data, trg_data):
        bucket_width = (max_len + num_buckets - 1) // num_buckets
        bucket_id = tf.size(src_data)  // bucket_width
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

      def key_func_scr(src_data, scr_data, trg_data):
        bucket_width = (max_len + num_buckets - 1) // num_buckets
        bucket_id = tf.size(src_data)  // bucket_width
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

      def reduce_func(unused_key, windowed_data):
        return windowed_data.padded_batch(batch_size_words, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None])))

      def reduce_func_scr(unused_key, windowed_data):
        return windowed_data.padded_batch(batch_size_words, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None])))

      def window_size_func(key):
        bucket_width = (max_len + num_buckets - 1) // num_buckets
        key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = (num_gpus * batch_size_words // (key * bucket_width))
        return tf.to_int64(size)

      if use_user_lang_map:
        src_trg_dataset = src_trg_dataset.filter(lambda src, scr, trg: tf.size(src)<=max_len)
        src_trg_dataset = src_trg_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func_scr, reduce_func=reduce_func_scr, window_size_func=window_size_func))
      else:
        src_trg_dataset = src_trg_dataset.filter(lambda src, trg: tf.size(src)<=max_len)
        src_trg_dataset = src_trg_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func))

    else:
        # debug model when batchsize = 1 in eval mode
        padded_shapes = ([None],[None])
        if use_user_lang_map: padded_shapes = ([None],[None],[None])
        src_trg_dataset = src_trg_dataset.padded_batch(batch_size*num_gpus, padded_shapes=padded_shapes)

    if use_user_lang_map:
      src_trg_dataset = src_trg_dataset.map(
            lambda src, scr, trg: ({"src":src, "scr":scr}, trg))
            #num_parallel_calls=10).prefetch(1000000)

    #src_trg_dataset = src_trg_dataset.repeat()
    iterator = src_trg_dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features,labels = iterator.get_next()
    return features,labels