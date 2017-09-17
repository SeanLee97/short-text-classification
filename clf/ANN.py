# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
一个简单的人造神经网络ANN
@author Sean
@last modified 2017.09.06 17:14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ANN(nn.Module):
	def __init__(self, **kwargs):
		super(ANN, self).__init__()
		self.input_size = kwargs['input_size']
		self.hidden_size = kwargs['hidden_size']
		self.output_size = kwargs['output_size']
		if 'dropout_p' in kwargs:
			self.dropout_p = kwargs['dropout_p']
		else:
			self.dropout_p = 0.1
		# 全连接层输出
		self.i2h = nn.Linear(self.input_size, self.hidden_size) # 输入->隐藏层
		self.h2o = nn.Linear(self.hidden_size, self.output_size) # hidden -> output
		self.softmax = nn.LogSoftmax()
		self.dropout = nn.Dropout(self.dropout_p)
	
	def forward(self, inp):
		inp = F.relu(self.i2h(inp))
		inp = self.dropout(inp)
		out = self.softmax(self.h2o(inp))
		return out
