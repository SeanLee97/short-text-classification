# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
BiGRU
@author Sean
@last modified 2017.09.06 17:14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class BiGRU(nn.Module):
	def __init__(self, **kwargs):
		super(BiGRU, self).__init__()
		self.input_size = kwargs['input_size']
		self.hidden_size = kwargs['hidden_size']
		self.output_size = kwargs['output_size']
		if 'embed_size' in kwargs:
			self.embed_size = kwargs['embed_size']
		else:
			self.embed_size = kwargs['hidden_size']
		if 'batch_size' in kwargs:
			self.batch_size = kwargs['batch_size']
		else:
			self.batch_size = 1
		if 'n_layers' in kwargs:
			self.n_layers = kwargs['n_layers']
		else: 
			self.n_layers = 1
		if 'dropout' in kwargs:
			self.dropout = kwargs['dropout']
		else:
			self.dropout = 0.1
		
		self.embed = nn.Embedding(self.input_size, self.embed_size)
		if 'word_embedding' in kwargs:
			# 使用预训练的embedding
			pretrained_weight = torch.from_numpy(kwargs['word_embedding'])
			self.embed.weight.data.copy_(pretrained_weight)
		
		self.bigru = nn.GRU(self.embed_size, self.hidden_size, dropout=self.dropout, num_layers=self.n_layers, bidirectional=True)
		# 双向GRU的 num_dicrections = 2
		# 全连接层输出
		self.h2o = nn.Linear(self.hidden_size * 2, self.output_size)
		self.hidden = self.init_hidden()
		self.dropout = nn.Dropout(self.dropout)

	def init_hidden(self):
		return Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))
	
	def forward(self, inp, inp_lens, hidden=None):
		embed = self.embed(inp) # N x batch_size x embed_size
		embed = self.dropout(embed)
		inp = embed.view(len(inp), embed.size(1), -1) 
		# pack
		inp = pack_padded_sequence(inp, inp_lens) 
		# gru的inp要求(seq_len, batch_size, *) 故在此reshape
		gru_out, hn = self.bigru(inp, hidden)
		# pad
		gru_out, output_lens = pad_packed_sequence(gru_out)
		#gru_out = gru_out[:, :, :self.hidden_size] + gru_out[:, : ,self.hidden_size:]

		# gru_out shape -> (n_layers, batch_size, 2*hidden_size) 
		gru_out = torch.transpose(gru_out, 0, 1)  # (batch_size, n_layers, 2*hidden_size)
		gru_out = torch.transpose(gru_out, 1, 2)  # (batch_size, 2*hidden_size, n_layers)

		gru_out = F.tanh(gru_out)
		# max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
		# kernerl_size = (2*n_layers, 2*n_layers)
		gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
		# gru_out shape -> input_size x 2*hidden_size
		gru_out = F.tanh(gru_out)
		y = self.h2o(gru_out)
		#print('y size', y.size())
		return y
