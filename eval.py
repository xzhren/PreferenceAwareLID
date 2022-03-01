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
from transformer import transformer_model_fn as model_fn
from data_reader import input_fn
import time
import tensorflow as tf
import numpy as np
import os, errno
import subprocess
import re
from functools import reduce

tf.logging.set_verbosity(tf.logging.INFO)

def cal_acc(output_file, dev_src, dev_trg, acc_log, trg_vocab):
    acc_lt = []

    with tf.gfile.GFile(output_file, 'r') as fout, tf.gfile.GFile(output_file+".res", 'w') as fres:
        hypotheses = fout.readlines()
        print('Num of decoded sentence:%d' % len(hypotheses))
        
        with tf.gfile.GFile(dev_src, 'r') as fdevsrc:
            try:
               devsrcs = []
               for i,line in enumerate(fdevsrc):
                 devsrcs.append(line)
            except:
               print("error:",i,line)
        with tf.gfile.GFile(dev_trg, 'r') as fdevtrg:
            devtrgs = fdevtrg.readlines()
        
        acc_counter, acc_counter_total = 0, 0
        acc_counter_lt = [0] * len(trg_vocab)
        acc_counter_all = [0] * len(trg_vocab)
        for hypothes, src, trg in zip(hypotheses, devsrcs, devtrgs):
            hypothes, src, trg = hypothes.strip(), src.strip(), trg.strip()
            if trg not in trg_vocab: continue
            trg_index = trg_vocab.index(trg)
            acc_counter_total += 1
            acc_counter_all[trg_index] += 1
            print(src+"\t"+trg+"\t"+hypothes, file=fres)
            if hypothes.split("\t")[0] == trg:
                acc_counter += 1
                acc_counter_lt[trg_index] += 1
        acc_lt = [acc_counter / acc_counter_total]
        acc_lt.extend([r/(a+0.0001) for r, a in zip(acc_counter_lt, acc_counter_all)])

    with tf.gfile.GFile(acc_log, 'a') as f:
        bleu = '%s' % ('\t'.join(format(x*100, "0.2f") for x in acc_lt))
        print('ACC: %s' % bleu)
        print(bleu, file=f)
    return bleu


def get_dict_from_collection(collection_name):
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  keys = tf.get_collection(key_collection)
  values = tf.get_collection(value_collection)
  return dict(zip(keys, values))

def extract_batches(tensors):
  if not isinstance(tensors, dict):
    for tensor in tensors:
      yield tensor
  else:
    batch_size = None
    for value in tensors.values():
      batch_size = batch_size or value.shape[0]
    for b in range(batch_size):
      yield {
          key: value[b] for key, value in tensors.items()
      }

class SaveEvaluationPredictionHook(tf.train.SessionRunHook):
  def __init__(self, output_file, dev_src, dev_trg, trg_vocab_file, acc_log):
    self._output_file = output_file
    self._dev_src = dev_src
    self._dev_trg = dev_trg
    self._acc_log = acc_log
    trg_rvocab = [w.strip() for w in open(trg_vocab_file)]
    self._trg_vocab = trg_rvocab
    self.devsrcs = [line.strip() for line in open(dev_src, "r", encoding='utf-8', errors='ignore').readlines()]
    self.devtrgs = [line.strip() for line in open(dev_trg, "r", encoding='utf-8', errors='ignore').readlines()]
    self.dev_index = -1

  def begin(self):
    self._predictions = get_dict_from_collection("predictions")
    self._features = tf.get_collection("features")
    self._labels = tf.get_collection("labels")
    if use_user_lang_map: self._contexts = tf.get_collection("contexts")
    self._global_step = tf.train.get_global_step()
    self.start_time = time.mktime(time.localtime())
    self.count = 0

  def before_run(self, run_context):
    if use_user_lang_map: return tf.train.SessionRunArgs([self._predictions, self._global_step, self._features, self._labels, self._contexts])
    else: return tf.train.SessionRunArgs([self._predictions, self._global_step, self._features, self._labels])

  def after_run(self, run_context, run_values):
    if use_user_lang_map:  predictions, current_step, features, labels, contexts = run_values.results
    else: predictions, current_step, features, labels = run_values.results
    self._output_path = "{}.{}".format(self._output_file, current_step)

    if self.count % 10000 == 0: tf.logging.info("eval sample:"+str(self.count))
    with open(self._output_path,'a') as output_file:
        for index, prediction in enumerate(extract_batches(predictions)):
            self.count += 1
            self.dev_index += 1
            prob = prediction['predict_score']
            prob_index = np.argsort(prob)[::-1]
            prob_index_lang = [self._trg_vocab[item] for item in prob_index]
            prob_value = np.sort(prob)[::-1]
            prob_index_lang = prob_index_lang[:3]
            prob_value = prob_value[:3]
            prob_str = "\t".join(['%s\t%.2f'%(i,v*100) for i,v in zip(prob_index_lang, prob_value)])
            print(prob_str, file=output_file)
            
            if showinfos:
                if use_user_lang_map: feature, context = features[0], contexts[0]
                else: feature = features[0]
                self.dev_index = self.dev_index % len(self.devsrcs)
                devsrc, devtrg = self.devsrcs[self.dev_index], self.devtrgs[self.dev_index]
                if devtrg != prob_index_lang[0]:
                    tf.logging.info("text:"+devsrc+" lang:"+devtrg)
                    tf.logging.info("features:"+" ".join([str(item) for item in feature[index]]))
                    if use_user_lang_map: tf.logging.info("context:"+" ".join([str(item) for item in context[index]]))
                    tf.logging.info("predict:"+prob_str)

  def end(self, session):
    _ = self.count    
    end_time = time.mktime(time.localtime())
    dur_sec = end_time - self.start_time
    tf.logging.info("Evaluation speed "+str((_+1)/dur_sec)+"e/s:"+str(dur_sec*1000/(_+1))+"ms/e")
    tf.logging.info("Evaluation predictions saved to %s", self._output_path)
    score = cal_acc(self._output_path, self._dev_src, self._dev_trg, self._acc_log, self._trg_vocab)
    tf.logging.info("accuracy: %s", score[:5])

def cvt(checkpoint_dir,output_dir):
    model_name = "NmtModel"
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            var_list = []
            var_sum = 0
            for var_name, var_shape in tf.contrib.framework.list_variables(checkpoint_dir):
                if 'ExponentialMovingAverage' in var_name or 'global_step' in var_name:
                    tf.logging.info("var_name: %s:%s", var_name, var_shape)
                    if 'ExponentialMovingAverage' in var_name: var_sum += reduce(lambda x,y:x*y, var_shape) 
                    var_value = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
                    var_name = var_name.replace('/'+model_name+'/', '/').replace('/'+model_name+'/', '/').replace('/ExponentialMovingAverage','')
                    if var_name == 'global_step':
                        step = str(var_value)
                    var = tf.Variable(var_value, name=var_name)
                    var_list.append(var)
            tf.logging.info("Params sum is %d", var_sum)
            saver = tf.train.Saver()
            sess.run(tf.variables_initializer(var_list))
            saver.save(sess, output_dir+'/model.ckpt-'+step)
            sess.close()

def sorted_dir(folder):
    def getmtime(name):
        path = os.path.join(folder, name)
        return os.path.getmtime(path)

    li = [f for f in os.listdir(folder) if '.index' in f]
    return sorted(li, key=getmtime, reverse=False)

def get_last_step(folder):
    def getmtime(name):
        return os.path.getmtime(dev_out+'.'+name)

    li = [re.sub(r'^.*?(\d+)$', r'\1',f) for f in os.listdir(folder) if re.match(r'.*\.\d+$',f)]
    if li == []:
        return '1'
    return sorted(li, key=getmtime, reverse=True)[0]

def main(_):
    for k in params.keys():
        if 'dropout' in k:
            params[k] = 0.0
    eval_input_fn = lambda: input_fn(
        dev_src,
        dev_trg,
        vocab_src,
        vocab_trg,
        ulid_file=dev_ulid,
        batch_size=params["decode_batch_size"],
        is_train=False,
        use_script_embedding=None,
        use_word_embedding=None,
        script_vocab_file=None,
        src_word_vocab_file=None
    )
    eval_hooks = []
    eval_hooks.append(SaveEvaluationPredictionHook(dev_out,dev_src,dev_trg,vocab_trg,acc_log))

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session_config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True, log_device_placement=False)
    eval_model_dir = model_dir.rstrip('\/')
    if params['ema_decay'] > 0.0:
        eval_model_dir = model_dir.rstrip('\/')+'/ema'
    transformer = tf.estimator.Estimator(model_fn=model_fn, model_dir=eval_model_dir, params=params, config=tf.estimator.RunConfig(session_config=session_config))
    try:
        os.makedirs(os.path.dirname(dev_out))
    except OSError as e:
        if e.errno != errno.EEXIST:
            tf.logging.info(e)
            raise

    if params['continuous_eval'] == True:
        current = 1
        if params["eval_from_step"] > 0:
            current = params["eval_from_step"] - 1
        while True:
            ckpts = sorted_dir(model_dir)
            for ckpt in ckpts:
                iteration = int(re.sub(r'^.*?(\d+).*', r'\1', ckpt))
                if iteration > current and os.path.isfile(model_dir.rstrip('\/')+'/'+ckpt):
                    if params['ema_decay'] > 0.0:
                        cvt(model_dir.rstrip('\/')+'/'+ckpt.replace('.index',''),eval_model_dir)
                    else:
                        with open(model_dir.rstrip('\/')+'/checkpoint') as ck:
                            old=ck.readline()
                            keep=ck.read()
                        with open(model_dir.rstrip('\/')+'/checkpoint', 'w') as newck:
                            print >> newck, 'model_checkpoint_path: \"'+ckpt.replace('.index','')+'\"'
                            print >> newck, keep
                            
                    try:
                        os.remove(dev_out+'.'+str(iteration))
                    except OSError:
                        pass
                    tf.logging.info("start evaluating...")

                    normal_bleu_lines = []
                    if tf.gfile.Exists(acc_log):
                        with tf.gfile.GFile(acc_log, 'r') as f:
                            bleu_lines = f.readlines()
                            for i in bleu_lines:
                                normal_bleu_lines.append(i)
                    # add new steps first
                    normal_bleu_lines.append('%s\t' % iteration)
                    with tf.gfile.GFile(acc_log, 'w') as f:
                        f.write(''.join(normal_bleu_lines))

                    transformer.evaluate(eval_input_fn, hooks=eval_hooks)
                    # except:
                    #     tf.logging.info( "Evaluation of checkpoint %s failed!" % ckpt)
                    current = iteration
            time.sleep(5)
    else:
        ckpt = open(model_dir.rstrip('\/')+'/checkpoint').readline().strip()
        ckpt = re.sub(r'^.*\: \"(.*?)\"', r'\1.index', ckpt)
        if params['ema_decay'] > 0.0:
            cvt(model_dir.rstrip('\/')+'/'+ckpt.replace('.index',''),eval_model_dir)
        transformer.evaluate(eval_input_fn, hooks=eval_hooks)

if __name__ == '__main__':
    tf.app.run()

