# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import glob, os
import jieba
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from clf.ANN import ANN
N_EPOCHS = 1000
LEARNING_RATE = 0.01
HIDDEN_SIZE = 128


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

def init_data(data, category):
	pairs = []
	categories = []
	lang = Lang()
	for i in range(len(data)):
		sentence_list = list(data[i])
		pairs.append({'category': category[i], 'sentence': sentence_list})
		if category[i] not in categories:
			categories.append(category[i])
		lang.index_words(sentence_list)
	return pairs, categories, lang

def load_dataset(corpus_dir='./corpus/data/', char_level=False):
	datas = []
	labels = []
	all_labels = []
	lang = Lang()
	for filename in os.listdir(corpus_dir):
		with open(os.path.join(corpus_dir, filename), 'r') as f:
			for line in f.readlines():
				line = line.strip().replace(' ', '')
				if len(line) == 0:
					continue
				if char_level:
					sentence_list = list(line)
				else:
					sentence_list = jieba.lcut(line)

				label = filename.split('.txt')[0]
				lang.index_words(sentence_list)	
				datas.append(sentence_list)
				labels.append(label)
				if label not in all_labels:
					all_labels.append(label)
	return datas, labels

def save_model(model, epoch, max_keep=5):
	if not os.path.exists('./runtime/ann_model'):
		os.makedirs('./runtime/ann_model')
	f_list = glob.glob(os.path.join('./runtime/ann_model', 'model') + '-*.ckpt')
	if len(f_list) >= max_keep + 2:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
		for f in to_delete:
			os.remove(f)
	name = 'model-{}.ckpt'.format(epoch)
	file_path = os.path.join('./runtime/ann_model', name)
	torch.save(model.state_dict(), file_path)

def load_previous_model(model):
	f_list = glob.glob(os.path.join('./runtime/ann_model', 'model') + '-*.ckpt')
	start_epoch = 1
	if len(f_list) >= 1:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		last_checkpoint = f_list[np.argmax(epoch_list)]
		if os.path.exists(last_checkpoint):
			print('load from {}'.format(last_checkpoint))
			model.load_state_dict(torch.load(last_checkpoint))
			start_epoch = np.max(epoch_list)
	return model, start_epoch

def sentence2tensor(data_loader, sentence_list):
	tensor = torch.zeros(1, data_loader['n_words'])
	for word in sentence_list:
		if word not in data_loader['lang'].word2index:
			continue
		tensor[0][data_loader['lang'].word2index[word]] = 1
	return tensor

def random_choice(l):
	return l[random.randint(0, len(l)-1)]

def random_train(data_loader):
	data = random_choice(data_loader['pairs'])
	category = data['category']
	sentence_list = data['sentence'] # input
	input_variable = Variable(sentence2tensor(data_loader, sentence_list)) # input tensor
	target_variable = Variable(torch.LongTensor([data_loader['categories'].index(category)])) # target tensor
	return category, sentence_list, input_variable, target_variable

def train(data_loader, model, loss_fn, optimizer, start_epoch=1):
	#print(data_loader)
	runing_loss = 0
	all_losses = []

	for epoch in range(start_epoch, N_EPOCHS+1):
		category, sentence_list, input_variable, target_variable = random_train(data_loader)

		optimizer.zero_grad()
		output_pred = model(input_variable)
		
		loss = loss_fn(output_pred, target_variable)
		loss.backward()
		optimizer.step()

		runing_loss += loss.data[0]
		if epoch % 500 == 0:
			save_model(model, epoch)
			k = 0
			topv, topi = output_pred.data.topk(1)
			pred = topi[0][0]
			out_index = target_variable.data.numpy()
			print('loss = ', runing_loss, '%', 
					'input=', ''.join(sentence_list), 
					'actual=', data_loader['categories'][out_index[0]], 
					'predict=', data_loader['categories'][pred])
			all_losses.append(runing_loss/500)
			runing_loss = 0
	return all_losses

def evaluate(data_loader, model, sentence_list):
	input_tensor = sentence2tensor(data_loader, sentence_list)
	input_variable = Variable(input_tensor)
	output = model(input_variable)
	top_v, top_i = output.data.topk(1)
	output_index = top_i[0][0]
	return data_loader['categories'][output_index], top_v[0][0], output_index

def predict(model, data_loader, data, category):
	train_data, test_data, train_label, test_label = train_test_split(data, category, test_size = 1/3)
	pairs, categories, lang = init_data(test_data, test_label)

	total = 0
	correct = 0.0
	for item in pairs:
		if len(item['sentence']) == 0:
			continue
		intent,top_value,top_index = evaluate(data_loader, model, item['sentence'])
		if item['category']==intent:
			correct += 1
		else:
			print('----------------------------------------')
			print('predict %s, actual %s' % (intent, item['category']))
			print(''.join(item['sentence']))
		total += 1
	print("测试集总共%d条记录, %d条分类失败, %d条分类正确, 准确率%f" % (total, total-correct, correct, correct/total))

def qa(model, data_loader):
	while True:
		sentence = input('user> ')
		sentence_list = jieba.lcut(sentence, cut_all=False)
		intent,top_value,top_index = evaluate(data_loader, model, sentence_list)
		print(intent)

def main():

	data, category = load_dataset()
	
	if not os.path.exists('./runtime/data_split.pkl'):
		train_data, test_data, train_label, test_label = train_test_split(data, category, test_size = 1/3)
		data_split = {
			'train_data': train_data,
			'test_data': test_data,
			'train_label': train_label,
			'test_label': test_label
		}
		with open('./runtime/data_split.pkl', 'wb') as f:
			pickle.dump(data_split, f, True)
	else:
		with open('./runtime/data_split.pkl', 'rb') as f:
			data_split = pickle.load(f)
		train_data, test_data, train_label, test_label = data_split['train_data'], data_split['test_data'], data_split['train_label'], data_split['test_label']

	pairs, categories, lang = init_data(train_data, train_label)
	
	input_size = lang.n_words
	output_size = len(categories)
	data_loader = {}
	data_loader['n_words'] = input_size
	data_loader['input_size'] = input_size
	data_loader['output_size'] = output_size
	data_loader['pairs'] = pairs
	data_loader['lang'] = lang
	data_loader['categories'] = categories

	model = ANN(input_size=input_size, hidden_size=HIDDEN_SIZE, output_size=output_size)
	model, start_epoch = load_previous_model(model)
	loss_fn = nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
	all_losses = train(data_loader, model, loss_fn, optimizer, start_epoch)
	predict(model, data_loader, test_data, test_label)
	'''
	plt.figure()
	plt.plot(all_losses)
	plt.show()
	'''
if __name__ == '__main__':
	main()
