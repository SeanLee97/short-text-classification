# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import numpy as np
import glob
import os
import random
from word2vec import Word2vec

class Lang(object):
	def __init__(self):
		self.word2index = {'PAD': 0}
		self.n_words = 1

	def index_words(self, sentence_list):
		for word in sentence_list:
			self.index_word(word)

	def index_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.n_words += 1

class Dataset(object):
	def __init__(self, datas, labels, lang, all_labels, batch_size=4, word2vec=False):
		self.datas = datas
		self.labels = labels
		self.lang = lang
		self.all_labels = all_labels
		self.batch_size = batch_size
		self.word_embedding = None
		if word2vec:
			self.word_embedding, self.embed_size = self.wordembed()

	def sentence_idx(self, sentence):
		inp = []
		for word in sentence:
			if word in self.lang.word2index:
				inp.append(self.lang.word2index[word])
			else:
				inp.append(0)
		return inp

	def sentence_pad(self, inp, max_len):
		inp = self.sentence_idx(inp)
		return inp + [0 for i in range(max_len-len(inp))]

	def evaluate_pad(self, inp):
		inp = self.sentence_idx(inp)
		inp_lens = [len(inp)]
		inp = [inp]
		return inp, inp_lens

	def wordembed(self):
		vocab = [k for k,v in self.lang.word2index.items()]
		#vocab = list(set(vocab))
		model = Word2vec()
		model.load()
		word_embedding, embed_size = model.word_vector(vocab)
		#word_embedding = word_embedding[:2]
		#print(word_embedding)
		word_embedding = np.array(word_embedding)
		#print(word_embedding)
		#print(word_embedding.shape, type(word_embedding))
		#exit()
		return word_embedding, embed_size

	def random_train(self):
		inputs = []
		inputs_len = []
		targets = []
		for i in range(self.batch_size):
			idx = random.randint(0, len(self.labels)-1)
			inputs.append(self.datas[idx])
			targets.append(self.all_labels.index(self.labels[idx]))
		inputs = sorted(inputs, key=lambda p: len(p), reverse=True)
		inputs_len = [len(x) for x in inputs]
		max_len = max(inputs_len)
		inputs_pad = [self.sentence_pad(item, max_len) for item in inputs]
		input_var = Variable(torch.LongTensor(inputs_pad))
		target_var = Variable(torch.LongTensor(targets))
		return input_var.transpose(0, 1), target_var, inputs_len

def save_model(model, epoch, save_path, max_keep=5):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	f_list = glob.glob(os.path.join(save_path, 'model') + '-*.ckpt')
	if len(f_list) >= max_keep + 2:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
		for f in to_delete:
			os.remove(f)
	name = 'model-{}.ckpt'.format(epoch)
	file_path = os.path.join(save_path, name)
	torch.save(model, file_path)

def load_previous_model(save_path):
	model = None
	f_list = glob.glob(os.path.join(save_path, 'model') + '-*.ckpt')
	start_epoch = 1
	if len(f_list) >= 1:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		last_checkpoint = f_list[np.argmax(epoch_list)]
		if os.path.exists(last_checkpoint):
			print('load from {}'.format(last_checkpoint))
			# CNN 不支持参数保存
			#model.load_state_dict(torch.load(last_checkpoint))
			model = torch.load(last_checkpoint)
			start_epoch = np.max(epoch_list)
	return model, start_epoch
