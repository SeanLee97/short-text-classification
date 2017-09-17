# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
TextCNN
@author Sean
@last modified 2017.09.06 17:14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np
import random

class TextCNN(nn.Module):
	def __init__(self, **kwargs):
		super(TextCNN, self).__init__()
		self.input_size = kwargs['input_size']
		self.hidden_size = kwargs['hidden_size']
		self.output_size = kwargs['output_size']
		if 'kernel_num' in kwargs:
			self.kernel_num = kwargs['kernel_num']
		else:
			self.kernel_num = 256
		if 'kernel_sizes' in kwargs:
			self.kernel_sizes = kwargs['kernel_sizes']
		else:
			self.kernel_sizes = [1, 2, 3, 4]
		if 'embed_size' in kwargs:
			self.embed_size = kwargs['embed_size']
		else:
			self.embed_size = kwargs['hidden_size']		
		if 'dropout' in kwargs:
			self.dropout = kwargs['dropout']
		else:
			self.dropout = 0.1
		if 'wide_conv' in kwargs:
			self.wide_conv = kwargs['wide_conv']
		else:
			self.wide_conv = False
		if 'init_weight' in kwargs:
			self.init_weight = kwargs['init_weight']
		else:
			self.init_weight = False
		if 'init_weight_value' in kwargs:
			self.init_weight_value = kwargs['init_weight_value']
		else:
			self.init_weight_value = 2.0		
		if 'batch_normal' in kwargs:
			self.batch_normal = kwargs['batch_normal']
		else:
			self.batch_normal = False
		if 'batch_normal_momentum' in kwargs:
			self.batch_normal_momentum
		else:
			self.batch_normal_momentum = 0.1
		if 'batch_normal_affine' in kwargs:
			self.batch_normal_affine = kwargs['batch_normal_affine']
		else:
			self.batch_normal_affine = False

		Ci = 1	# input channels, 处理文本,一层通道
		Co = self.kernel_num	# output channel
		Ks = self.kernel_sizes	# list
		
		if 'max_norm' in kwargs:
			self.embed = nn.Embedding(self.input_size, self.embed_size, max_norm=kwargs['max_norm'])
		else:
			self.embed = nn.Embedding(self.input_size, self.embed_size, scale_grad_by_freq=True)
		if 'word_embedding' in kwargs:
			pretrained_weight = torch.from_numpy(kwargs['word_embedding'])
			self.embed.weight.data.copy_(pretrained_weight)
			self.embed.weight.requires_grad = True
		if self.wide_conv is True:
			self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, self.embed_size), stride=(1, 1), padding=(K//2 ,0), dilation=1, bias=True) for K in Ks]
		else:
			self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, self.embed_size), bias=True) for K in Ks]
		if self.init_weight:
			for conv in self.convs1:
				init.xavier_normal(conv.weight.data, gain=np.sqrt(self.init_weight_value))
				fanin, fanout = self.cal_fanin_fanout(conv.weight.data)
				std = np.sqrt(self.init_weight_value) * np.sqrt(2.0 / (fanin+fanout))
				init.uniform(conv.bias, 0, 0)

		self.dropout = nn.Dropout(self.dropout)
		in_fea = len(Ks) * Co
		self.f1 = nn.Linear(in_fea, in_fea//2, bias=True)
		self.f2 = nn.Linear(in_fea//2, self.output_size, bias=True)
		if self.batch_normal:
			self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=self.batch_normal_momentum, affine=self.batch_normal_affine)
			self.f1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=self.batch_normal_momentum, affine=self.batch_normal_affine)
			self.f2_bn = nn.BatchNorm1d(num_features=self.output_size, momentum=self.batch_normal_momentum, affine=self.batch_normal_affine)

	def cal_fanin_fanout(self, tensor):
		dims = tensor.ndimension()
		if dims < 2:
			raise ValueError('fan in and fan out should be more than 2 dimensions')
		
		if dims == 2:
			fan_in = tensor.size(1)
			fan_out = tensor.size(0)
		else:
			num_inp_fmaps = tensor.size(1)
			num_out_fmaps = tensor.size(0)
			receptive_field_size = 1
			if tensor.dim() > 2:
				receptive_field_size = tensor[0][0].numel()
			fanin = num_inp_fmaps * receptive_field_size
			fanout = num_out_fmaps * receptive_field_size
		return fanin, fanout

	def forward(self, inp):
		inp = self.embed(inp) # N x W x embed_size -> seq_len x batch_size x embed_size
		inp = torch.transpose(inp, 0, 1) # batch_size x seq_len
		inp = inp.unsqueeze(1) # N x Ci x W x embed_size  -> Ci = 1 --> batch_size x 1 x seq_len x embed_size

		if self.batch_normal is True:
			inp = [self.convs1_bn(F.relu(conv(inp))).squeeze(3) for conv in self.convs1]	# [(batch_size, seq_len, Co), ...]*len(Ks)
			inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]	# [(N, Co), ...]*len(Ks)
		else:
			inp = [self.dropout(F.relu(conv(inp)).squeeze(3)) for conv in self.convs1]	# [(batch_size, seq_len, Co), ...]*len(Ks)
			inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]	# [(batch_size, seq_len), ...]*len(Ks)

		inp = torch.cat(inp, 1)
		inp = self.dropout(inp)	# (batch_size, len(Ks)*Co)
		if self.batch_normal is True:
			inp = self.f1_bn(self.f1(inp))
			logit = self.f2_bn(self.f2(F.relu(inp)))
		else:
			inp = self.f1(inp)
			logit = self.f2(F.relu(inp))
		return logit
