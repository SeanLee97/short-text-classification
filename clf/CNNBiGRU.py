# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
CNN+BiGRU
@author Sean
@last modified 2017.09.06 17:14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import numpy as np
import random

class CNNBiGRU(nn.Module):
	def __init__(self, **kwargs):
		super(CNNBiGRU, self).__init__()
		self.input_size = kwargs['input_size']
		self.hidden_size = kwargs['hidden_size']
		self.output_size = kwargs['output_size']
		if 'n_layers' in kwargs:
			self.n_layers = kwargs['n_layers']
		else:
			self.n_layers = 1
		if 'batch_size' in kwargs:
			self.batch_size = kwargs['batch_size']
		else:
			self.batch_size = 1
		if 'embed_size' in kwargs:
			self.embed_size = kwargs['embed_size']
		else:
			self.embed_size = kwargs['hidden_size']
		if 'kernel_num' in kwargs:
			self.kernel_num = kwargs['kernel_num']
		else:
			self.kernel_num = 256
		if 'kernel_sizes' in kwargs:
			self.kernel_sizes = kwargs['kernel_sizes']
		else:
			self.kernel_sizes = [1,2,3,4]
		if 'dropout' in kwargs:
			self.dropout = kwargs['dropout']
		else:
			self.dropout = 0.1

		Ci = 1
		Co = self.kernel_num
		Ks = self.kernel_sizes
		
		self.embed = nn.Embedding(self.input_size, self.embed_size)
		if 'word_embedding' in kwargs:
			pretrained_weight = torch.from_numpy(kwargs['word_embedding'])
			self.embed.weight.data.copy_(pretrained_weight)
		
		# conv2d卷积操作
		self.convs1 = [nn.Conv2d(Ci, Co, (K, self.embed_size), padding=(K//2, 0), stride=(1,1)) for K in Ks]
		
		# BiGRU num_directions = 2
		self.bigru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropout, bidirectional=True, bias=True)
		self.hidden = self.init_hidden()

		# Linear
		L = len(Ks)*Co + self.hidden_size*2  # num_directions = 2
		self.h2o1 = nn.Linear(L, L//2)
		self.h2o2 = nn.Linear(L//2, self.output_size)
		#self.h2o = nn.Linear(L, self.output_size)
		self.dropout = nn.Dropout(self.dropout)

	def init_hidden(self):
		# num_directions = 2
		return Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))

	def forward(self, inp, inp_lens, hidden=None):
		embed = self.embed(inp) # N x batch_size x embed_size
		embed = self.dropout(embed)
		inp = embed.view(len(inp), embed.size(1), -1) 
		cnn = inp
		cnn = torch.transpose(cnn, 0, 1) # (batch_size, seq_len, embed_size)
		cnn = cnn.unsqueeze(1) # (batch_size, 1, seq_len, embed_size)
		cnn = [F.relu(conv(cnn)).squeeze(3) for conv in self.convs1] # [(batch_size, seq_len, Co), ...]*len(Ks)
		# max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)最大值池化
		# kernel_size 池化窗口大小
		# input(samples, features, steps)
		# output(samples, features, down_steps) 
		cnn = [F.relu(F.max_pool1d(i, i.size(2))).squeeze(2) for i in cnn]	
		cnn = torch.cat(cnn, 1) # 在1轴拼接 (seq_len, Co*len(Ks))
		cnn = self.dropout(cnn)
		# BiGRU
		bigru = inp.view(len(inp), inp.size(1), -1) # seq_len x batch_size x embed_size
		bigru = pack_padded_sequence(bigru, inp_lens)
		bigru, hn = self.bigru(bigru, hidden) 
		bigru, output_lens = pad_packed_sequence(bigru)
		# bigru shape -> (seq_len, batch_size, 2*hidden_size)
		bigru = torch.transpose(bigru, 0, 1)	# (batch_size ,seq_len, 2*hidden_size)
		bigru = torch.transpose(bigru, 1, 2) 	# (batch_size, 2*hidden_size, seq_len)
		bigru = F.max_pool1d(bigru, bigru.size(2)).squeeze(2)	# (batch_size, 2*hidden_sze)
		bigru = F.relu(bigru)
		
		# CNN BiGRU
		cnn = torch.transpose(cnn, 0, 1) # (Co*len(Ks), batch_size)
		bigru = torch.transpose(bigru, 0, 1) #(2*hidden_size, batch_size)
		cnn_bigru = torch.cat((cnn, bigru), 0) # (Co*len(Ks) + 2*hidden_size, batch_size)
		cnn_bigru = torch.transpose(cnn_bigru, 0, 1) #(batch_size, Co*len(Ks)+2*hidden_size)
		cnn_bigru = self.h2o1(F.relu(cnn_bigru))
		logit = self.h2o2(F.relu(cnn_bigru))
		#logit = self.h2o(F.relu(cnn_bigru))
		return logit
