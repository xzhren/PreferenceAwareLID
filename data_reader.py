#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
#
# History:
# 2019.07.22. Be created by xingzhang.rxz. Used for language identification in CNN method.
# 2018.04.26. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor. 
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
# from tools.utils_unicodes import UNICODE_BLOCK
# from tools.mapper_src_bpe import BPE
#import imp
#imp.reload(sys)
#sys.setdefaultencoding('utf-8')
#use_subword_embedding = True
#src_bpe_vocab_file = "corpus_test/bpe.codes"
#src_bpe_vocab_file = "corpus104v4_500w_vocab/bpe.codes"
# ublock_utls = UNICODE_BLOCK()

# src_bpe = None
# if use_subword_embedding: 
#     src_codes = codecs.open(src_bpe_vocab_file, encoding='utf-8')
#     src_bpe = BPE(src_codes, "@@")

# def string_to_unicode_script_ids(str_line, ublock_utls):
#     unicode_script_ids = ublock_utls.get_str_ublock_id(str_line)
#     assert len(unicode_script_ids) == len(str_line)
#     return unicode_script_ids


def _replace_str(str_raw):
    return str_raw.replace("\xa0", "").replace("\u2028", "").replace("\x85", "").replace("\u2009", "").replace("\u202f", "")

def paste_text_file_to_chars(data_src, is_train):
  for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
    # tmpline = line.rstrip("\n").lstrip("").replace(" ", "")
    if is_train and lineid < start_examples: continue
    tmpline = line.lower().rstrip("\n").replace(" ", "")
    tmpline = _replace_str(tmpline)
    yield list(tmpline)

def paste_text_file_to_labels(data_src, is_train):
  for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
    if is_train and lineid < start_examples: continue
    # tmpline = line.rstrip("\n").lstrip("").replace(" ", "")
    # yield [tmpline]
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


# def paste_text_file_to_scripts(data_src, ublock_utls, is_train):
#   for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
#     if is_train and lineid < start_examples: continue
#     # tmpline = line.rstrip("\n").lstrip("").replace(" ", "")
#     tmpline = line.lower().rstrip("\n").replace(" ", "")
#     tmpline = _replace_str(tmpline)
#     scriptline = []
#     for item in list(tmpline):
#         if item in ["</S>", "<UNK>", "", ""]:
#           scriptline.append(ublock_utls.unkindex)
#         else:
#           scriptline.append(ublock_utls.get_char_ublock_id(item))
#     assert len(scriptline) == len(tmpline)
#     yield scriptline

# def paste_text_file_to_words(data_src, src_bpe, is_train):
#   #for line in tf.gfile.GFile(data_src,"r"):
# #   if use_subword_embedding: 
# #     src_codes = codecs.open(src_bpe_codes, encoding='utf-8')
# #     src_bpe = BPE(src_codes, "@@")
#     # print("bpe codes: ", src_bpe.bpe_codes)

#   for lineid, line in enumerate(open(data_src, "r", encoding='utf-8', errors='ignore')):
#     if is_train and lineid < start_examples: continue
#     # tmpline = line.rstrip("\n").lstrip("").replace(" ", "").replace("", " ")
#     tmpline = line.lower().rstrip("\n")
#     tmpline = _replace_str(tmpline)
#     res = ""
#     ch_stop_chars = "。、！？：；﹑•＂…‘’“”〝〞∕¦‖—　〈〉﹞﹝「」‹›〖〗】【»«』『〕〔》《﹐¸﹕︰﹔！¡？¿﹖﹌﹏﹋＇´ˊˋ―﹫︳︴¯＿￣﹢﹦﹤‐­˜﹟﹩﹠﹪﹡﹨﹍﹉﹎﹊ˇ︵︶︷︸︹︿﹀︺︽︾ˉ﹁﹂﹃﹄︻︼（）"
#     en_stop_chars = "!\"#$%&'()*+,-./:;<=>?@[]^_‘{|}~\\"
#     stop_chars = ch_stop_chars + en_stop_chars
#     for item in tmpline:
#         if item in stop_chars:
#             item = " "
#         res += item
#     # print(len(res.strip().split(" ")), res.strip().strip(""))
#     wlt = res.split(" ")
#     #wlt = res.split()
#     # words_inp = [""]
#     words_inp = []
#     for word in wlt:
#       if use_subword_embedding: 
#         subword = src_bpe.segment(word)
#         # print(word, subword)
#         for subitem in subword.split(" "):
#           words_inp.extend([subitem]*len(subitem.replace("@@", "")))
#       else:
#         words_inp.extend([word]*len(word))
#       words_inp.append("")
#     del words_inp[-1]
#     # if len(words_inp) != len(list(line.rstrip("\n").lstrip("").replace(" ", ""))):
#     # if len(words_inp) != len(list(_replace_str(line.lower().rstrip("\n")))):
#     if len(words_inp) != len(tmpline):
#       #print(len(wlt), wlt)
#       tf.logging.info("wlt: %s, wordlt: %s" % (wlt, words_inp))
#       tf.logging.info("error res data: len %d text %s" % (len(words_inp), words_inp))
#       tf.logging.info("error raw data: len %d text %s" % (len(list(_replace_str(line.lower().rstrip("\n")))), list(_replace_str(line.lower().rstrip("\n")))))
#       #tf.logging.info("error raw data: len %d text %s" % (len(list(line.rstrip("\n").lstrip("").replace(" ", ""))), line.rstrip("\n").lstrip("").split(" ")))
#     #assert len(words_inp) == len(line.rstrip("\n").split(" "))
#     # wordline = " ".join(words_inp)
#     # print(line.strip(), wordline)
#     # yield wordline
#     yield words_inp

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
    # if use_script_embedding:
      # key, value = [], []
      # ublock_utls = UNICODE_BLOCK()
      # for index, line in enumerate(tf.gfile.GFile(src_vocab_file,"r")):
      #   if line.strip() == "": continue
      #   vocabtmp = line.strip()
      #   key.append(vocabtmp)
      #   if vocabtmp in ["</S>", "<UNK>", "", ""]:
      #     value.append(ublock_utls.unkindex)
      #   else:
      #     value.append(ublock_utls.get_char_ublock_id(vocabtmp))
      # for index, line in enumerate(tf.gfile.GFile(script_vocab_file,"r")):
      #   info = line.strip().split("\t")
      #   if len(info) != 2: print(index, line.strip()); continue
      #   key.append(info[0])
      #   value.append(int(info[1]))
      # script_vocabs = list(zip(key, value))
      # tf.logging.info("load srcipt vocab:"+str(len(script_vocabs))+" key value:"+str(key[:100])+str(value[:100]))      
      # script_vocab = tf.contrib.lookup.HashTable(initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys=key, values=value, value_dtype=tf.int64),default_value=1)

    # src_dataset = tf.data.TextLineDataset(src_file)
    # trg_dataset = tf.data.TextLineDataset(trg_file)
    src_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_chars(data_src=src_file, is_train=is_train), tf.string, tf.TensorShape([None]))
    trg_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_labels(data_src=trg_file, is_train=is_train), tf.string, tf.TensorShape([None]))

    # if use_script_embedding:
    #   ublock_utls = UNICODE_BLOCK()
    #   src_script_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_scripts(data_src=src_file, ublock_utls=ublock_utls, is_train=is_train), tf.int64, tf.TensorShape([None]))
    #   src_trg_dataset = tf.data.Dataset.zip((src_dataset, src_script_dataset, trg_dataset))
    # elif use_word_embedding: 
    #   src_word_vocab = tf.contrib.lookup.index_table_from_file(src_word_vocab_file,default_value=1)
    #   src_word_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_words(data_src=src_file, src_bpe=src_bpe, is_train=is_train), tf.string, tf.TensorShape([None]))
    #   src_trg_dataset = tf.data.Dataset.zip((src_dataset, src_word_dataset, trg_dataset))
    # elif use_word_script_embedding:
    #   ublock_utls = UNICODE_BLOCK()
    #   src_script_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_scripts(data_src=src_file, ublock_utls=ublock_utls, is_train=is_train), tf.int64, tf.TensorShape([None]))
    #   src_word_vocab = tf.contrib.lookup.index_table_from_file(src_word_vocab_file,default_value=1)
    #   src_word_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_words(data_src=src_file, src_bpe=src_bpe, is_train=is_train), tf.string, tf.TensorShape([None]))
    #   src_trg_dataset = tf.data.Dataset.zip((src_dataset, src_script_dataset, src_word_dataset, trg_dataset))
    if use_user_lang_map:
      ulid_dataset = tf.data.Dataset.from_generator(lambda: paste_text_file_to_ulids(data_src=ulid_file, is_train=is_train), tf.float32, tf.TensorShape([None]))
      src_trg_dataset = tf.data.Dataset.zip((src_dataset, ulid_dataset, trg_dataset))       
    else:
      src_trg_dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    
    # char granulairty, add head tag, tail tag is equivalent to eos
    # if vocab_granularity == 'char':
    # if use_word_embedding:
    #     src_trg_dataset = src_trg_dataset.map(
    #             lambda src, srcw, trg: (tf.string_split([src], delimiter=' ').values, tf.string_split([srcw], delimiter=' ').values, [trg]), num_parallel_calls=10).prefetch(args.prefetch)
    # else:
    #     src_trg_dataset = src_trg_dataset.map(
    #             lambda src, trg: (tf.string_split([src], delimiter=' ').values, [trg]), num_parallel_calls=10).prefetch(args.prefetch)

    # else:
    #     src_trg_dataset = src_trg_dataset.map(
    #         lambda src, trg: ( tf.string_split([src]).values, tf.string_split([trg], delimiter=' ').values), # vocab granulairty
    #         num_parallel_calls=10).prefetch(args.prefetch)
    
    # if use_script_embedding or use_user_lang_map:
    if use_user_lang_map:
      src_trg_dataset = src_trg_dataset.map(
            # lambda src, trg: (src_vocab.lookup(src), script_vocab.lookup(src), trg_vocab.lookup(trg)),
            lambda src, srcs, trg: (src_vocab.lookup(src), srcs, trg_vocab.lookup(trg)),
            num_parallel_calls=10).prefetch(args.prefetch)
    # elif use_word_embedding:
    #   src_trg_dataset = src_trg_dataset.map(
    #         lambda src, srcw, trg: (src_vocab.lookup(src), src_word_vocab.lookup(srcw), trg_vocab.lookup(trg)),
    #         num_parallel_calls=10).prefetch(args.prefetch)
    # elif use_word_script_embedding:
    #    src_trg_dataset = src_trg_dataset.map(
    #         lambda src, srcs, srcw, trg: (src_vocab.lookup(src), srcs, src_word_vocab.lookup(srcw), trg_vocab.lookup(trg)),
    #         num_parallel_calls=10).prefetch(args.prefetch)
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
     
    #   def key_func_scr_two(src_data, scr1_data, scr2_data,  trg_data):
    #     bucket_width = (max_len + num_buckets - 1) // num_buckets
    #     bucket_id = tf.size(src_data)  // bucket_width
    #     return tf.to_int64(tf.minimum(num_buckets, bucket_id))

      def reduce_func(unused_key, windowed_data):
        return windowed_data.padded_batch(batch_size_words, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None])))

      def reduce_func_scr(unused_key, windowed_data):
        return windowed_data.padded_batch(batch_size_words, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None])))

    #   def reduce_func_scr_two(unused_key, windowed_data):
    #     return windowed_data.padded_batch(batch_size_words, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None])))

      def window_size_func(key):
        bucket_width = (max_len + num_buckets - 1) // num_buckets
        key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = (num_gpus * batch_size_words // (key * bucket_width))
        return tf.to_int64(size)

      if use_script_embedding or use_word_embedding or use_user_lang_map:
        src_trg_dataset = src_trg_dataset.filter(lambda src, scr, trg: tf.size(src)<=max_len)
        src_trg_dataset = src_trg_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func_scr, reduce_func=reduce_func_scr, window_size_func=window_size_func))
    #   elif use_word_script_embedding:
    #     src_trg_dataset = src_trg_dataset.filter(lambda src, scrs, srcw, trg: tf.size(src)<=max_len)
    #     src_trg_dataset = src_trg_dataset.apply(
    #       tf.contrib.data.group_by_window(
    #           key_func=key_func_scr_two, reduce_func=reduce_func_scr_two, window_size_func=window_size_func))
      else:
        src_trg_dataset = src_trg_dataset.filter(lambda src, trg: tf.size(src)<=max_len)
        src_trg_dataset = src_trg_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func))

    else:

        # debug test cnn when batchsize = 1 in eval mode
        #src_trg_dataset = src_trg_dataset.filter(lambda src, trg: tf.size(src)>=5)
        padded_shapes = ([None],[None])
        if use_user_lang_map: padded_shapes = ([None],[None],[None])
        # if use_script_embedding or use_word_embedding or use_user_lang_map:  padded_shapes = ([None],[None],[None])
        # if use_word_script_embedding: padded_shapes = ([None],[None],[None],[None])
        src_trg_dataset = src_trg_dataset.padded_batch(batch_size*num_gpus, padded_shapes=padded_shapes)

    if use_user_lang_map:
    # if use_script_embedding or use_word_embedding or use_user_lang_map:
      src_trg_dataset = src_trg_dataset.map(
            #lambda src, scr, trg: (tf.stack([src,scr]), trg))
            lambda src, scr, trg: ({"src":src, "scr":scr}, trg))
            #num_parallel_calls=10).prefetch(1000000)
    # if use_word_script_embedding:
    #   src_trg_dataset = src_trg_dataset.map(
    #         lambda src, scrs, scrw, trg: ({"src":src, "scr1":scrs, "scr2": scrw}, trg))

    #src_trg_dataset = src_trg_dataset.repeat()
    iterator = src_trg_dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features,labels = iterator.get_next()
    return features,labels


# if __name__=="__main__":
#     # train_src = "./data/train.src"
#     # train_trg = "./data/train.trg"
#     # vocab_src = "./data/vocab.txt"
#     # vocab_trg = "./data/label.txt"
#     # train_input_fn = input_fn(
#     #         train_src,
#     #         train_trg,
#     #         vocab_src,
#     #         vocab_trg,
#     #         100,
#     #         100,
#     #         4, 
#     #         is_train=True
#     # )
    
#     #sess = tf.Session()
#     #sess.run(train_input_fn)

#     # import time
#     # for index, line in enumerate(paste_text_file_to_ulids("corpus21_ulid/train.ulid", is_train=True)):
#     # for index, line in enumerate(paste_text_file_to_ulids("corpus21_ulid/test_ulid/test.ulid", is_train=False)):
#     #     print(index, line)
#     #     if index > 100: break

#     for index, (line, label, char, word, script) in  enumerate(zip(open("corpus200_200w_1111/train.src"), 
#         paste_text_file_to_labels("corpus200_200w_1111/train.trg", False),
#         paste_text_file_to_chars("corpus200_200w_1111/train.src", False),
#         paste_text_file_to_words("corpus200_200w_1111/train.src", None, False),
#         # paste_text_file_to_scripts("corpus200_200w_1111/train.src", ublock_utls = UNICODE_BLOCK(), is_train=False)):
#         paste_text_file_to_words("corpus200_200w_1111/train.src", None, False)):
#         print()
#         print("lines:", index, line.strip())
#         print("label:", label)
#         print("char inputs:", char, len(char))
#         print("word inputs:", word, len(word))
#         print("script inputs:", script, len(script))
#         if index > 100: break
#         # if index < 16400000: continue
#         if index % 100000 == 0: print(index)

#     # print("word inputs:")
#     # for index, line in enumerate(paste_text_file_to_words("corpus104v6_500w/train.src")):
#     # #for index, line in enumerate(paste_text_file_to_words("corpus_test/train.109m.src")):
#     # #   if index > 630 and index < 640:
#     #   if index < 10:
#     #       print(index, line)
#     #   else: break
#     #   # print(index, end=" ")
#     # #   if index > 640: break
#     # #   if index % 100 == 0: print(index)

#     # print("script inputs:")
#     # for index, line in enumerate(paste_text_file_to_scripts("corpus104v6_500w/train.src", ublock_utls = UNICODE_BLOCK())):
#     # #for index, line in enumerate(paste_text_file_to_words("corpus_test/train.109m.src")):
#     # #   if index > 630 and index < 640:
#     #   if index < 10:
#     #       print(index, line)
#     #   else: break
#     #   # print(index, end=" ")
#     # #   if index > 640: break
#     # #   if index % 100 == 0: print(index)

#     # print("char inputs:")
#     # for index, line in enumerate(paste_text_file_to_chars("corpus104v6_500w/train.src")):
#     # #for index, line in enumerate(paste_text_file_to_words("corpus_test/train.109m.src")):
#     # #   if index > 630 and index < 640:
#     #   if index < 10:
#     #       print(index, line)
#     #   else: break
#     #   # print(index, end=" ")
#     # #   if index > 640: break
#     # #   if index % 100 == 0: print(index)

#     # print("label inputs:")
#     # for index, line in enumerate(paste_text_file_to_labels("corpus104v6_500w/train.trg")):
#     # #for index, line in enumerate(paste_text_file_to_words("corpus_test/train.109m.src")):
#     # #   if index > 630 and index < 640:
#     #   if index < 10:
#     #       print(index, line)
#     #   else: break
#     #   # print(index, end=" ")
#     # #   if index > 640: break
#     # #   if index % 100 == 0: print(index)
