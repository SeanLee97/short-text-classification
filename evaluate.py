# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import jieba
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import fasttext

from model_utils import *
from clf.CNNBiGRU import CNNBiGRU
from clf.BiGRU import BiGRU
from clf.TextCNN import TextCNN


def tokenizer(inp):
	return jieba.lcut(inp)

def get_accuracy(path='./runtime/accuracy.pkl'):
	if not os.path.exists(path):
		raise ValueError('not exists accuracy.pkl')
	with open(path, 'rb') as f:
		dic = pickle.load(f)
	return dic

def save_accuracy(dic, path='./runtime/accuracy.pkl'):
	if os.path.exists(path):
		dic2 = get_accuracy()
		dic = dict(dic2, **dic)
	with open(path, 'wb') as f:
		pickle.dump(dic, f, True)

def load_file(corpus_dir='./corpus/data/', char_level=False):
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
	return datas, labels, lang, all_labels

def load_data(dataset_path='./runtime/data_split.pkl'):
	if not os.path.exists(dataset_path):
		datas, labels, lang, all_labels = load_file()
		train_data, test_data, train_label, test_label = train_test_split(datas, labels, test_size = 1/3)
		data_split = {
			'lang': lang,
			'all_labels': all_labels,
			'train_data': train_data,
			'test_data': test_data,
			'train_label': train_label,
			'test_label': test_label
		}
		with open(dataset_path, 'wb') as f:
			pickle.dump(data_split, f, True)
	else:
		with open(dataset_path, 'rb') as f:
			data_split = pickle.load(f)
		lang, all_labels = data_split['lang'], data_split['all_labels']
		train_data, test_data, train_label, test_label = data_split['train_data'], data_split['test_data'], data_split['train_label'], data_split['test_label']
	return train_data, test_data, train_label, test_label, lang, all_labels
	

def svm_train(**kwargs):
	train_data, test_data, train_label, test_label, _, _ = load_data()
	train_data = [''.join(x) for x in train_data]
	vector = TfidfVectorizer(binary=False, tokenizer=tokenizer)
	train_vec = vector.fit_transform(train_data)
	estimator = Pipeline([('clf', LinearSVC())])
	param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
	train_scores, test_scores = validation_curve(estimator=estimator, X=train_vec, y=train_label, param_name='clf__C', param_range=param_range, cv=10)

	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)

	fig = plt.figure(figsize=(15, 6))
	plt.plot(param_range, train_mean,color='blue', marker='o', markersize=5, label='training accuracy')
	plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
	plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
	plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')
	plt.grid()
	plt.xscale('log')
	plt.legend(loc='lower right')
	plt.xlabel('Parameter C')
	plt.ylabel('Accuracy')
	plt.ylim([0.8, 1.0])
	plt.show()

def svm_test(**kwargs):
	if 'C' not in kwargs:
		raise ValueError('svm_test(C=x)')
	train_data, test_data, train_label, test_label, _, _ = load_data()
	train_data = [''.join(x) for x in train_data]
	vector = TfidfVectorizer(binary=False, tokenizer=tokenizer)
	train_vec = vector.fit_transform(train_data)
	clf = LinearSVC(C=kwargs['C'])
	clf.fit(train_vec, train_label)

	correct_num = 0
	test_vec = [vector.transform([''.join(x)]) for x in test_data]
	for i in range(len(test_vec)):
		pred = clf.predict(test_vec[i])
		if pred[0] == test_label[i]:
			correct_num += 1
	accuracy = correct_num/len(test_data)
	save_accuracy({'svm': accuracy})
	print('svm accuracy: ', accuracy)

def fasttext_clf(kwargs):
	'''
	args = {
		'lr_rate': 0.001,
		'n_epochs': 3,
		'train_path': './runtime/fasttext/train.txt',
		'test_path': './runtime/fasttext/test.txt',
		'model_path': './runtime/fasttext/model',
	}
	'''

	if not os.path.exists(kwargs['train_path']) or not os.path.exists(kwargs['test_path']):
		train_data, test_data, train_label, test_label, lang, all_labels = load_data()
		normal_labels = list(set(train_label) | set(test_label))

		os.makedirs(os.path.join('.', '/'.join(kwargs['train_path'].split('/')[0:-1])), exist_ok=True)
		os.makedirs(os.path.join('.', '/'.join(kwargs['test_path'].split('/')[0:-1])), exist_ok=True)
		with open(kwargs['train_path'], 'w') as f1, open(kwargs['test_path'], 'w') as f2:
			lines = ''
			for i in range(len(train_data)):
				lines += ' '.join(train_data[i]) + '\t__label__' + str(normal_labels.index(train_label[i])) + '\n'
			f1.writelines(lines)

			lines = ''
			for i in range(len(test_data)):
				lines += ' '.join(test_data[i]) + '\t__label__' + str(normal_labels.index(test_label[i])) + '\n'
			f2.writelines(lines)
		clf = fasttext.supervised(kwargs['train_path'], kwargs['model_path'], label_prefix="__label__",  epoch=kwargs['n_epochs'])
	else:
		clf = fasttext.load_model(kwargs['model_path']+'.bin', label_prefix="__label__")

	result = clf.test(kwargs['test_path'])
	save_accuracy({'fasttext': result.precision})
	print('fasttext accuracy', result.precision)


def trimodel_train(kwargs):
	'''
	args = {
		'n_epochs': 3,
		'batch_size': 1,
		'lr_rate': 0.001,
		'hidden_size': 64,
		'word2vec': False,
		'save_path': './runtime/evaluate_model',
		'evaluate_every': 300
	}
	'''

	train_data, test_data, train_label, test_label, lang, all_labels = load_data()
	input_size = lang.n_words
	output_size = len(all_labels)

	model1, start_epoch1 = load_previous_model(save_path = kwargs['save_path'] + '1')
	model2, start_epoch2 = load_previous_model(save_path = kwargs['save_path'] + '2')
	model3, start_epoch3 = load_previous_model(save_path = kwargs['save_path'] + '3')

	if 'word2vec' in kwargs and kwargs['word2vec'] is True:
		dataset = Dataset(train_data, train_label, lang, all_labels, batch_size=kwargs['batch_size'], word2vec=True)
		if model1 == None:
			model1 = TextCNN(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
				word_embedding=dataset.word_embedding, embed_size=dataset.embed_size
			)
		if model2 == None:
			model2 = BiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
				word_embedding=dataset.word_embedding, embed_size=dataset.embed_size
			)
		if model3 == None:
			model3 = CNNBiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], output_size = output_size, batch_size=kwargs['batch_size'], 
				word_embedding=dataset.word_embedding, embed_size=dataset.embed_size
			)
	else:
		dataset = Dataset(train_data, train_label, lang, all_labels, batch_size=kwargs['batch_size'])
		if model1 == None:
			model1 = TextCNN(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
			)
		if model2 == None:
			model2 = BiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
			)
		if model3 == None:
			model3 = CNNBiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], output_size = output_size, batch_size=kwargs['batch_size']
			)

	optimizer1 = torch.optim.Adam(model1.parameters(), lr = kwargs['lr_rate'])
	optimizer2 = torch.optim.Adam(model2.parameters(), lr = kwargs['lr_rate'])
	optimizer3 = torch.optim.Adam(model3.parameters(), lr = kwargs['lr_rate'])

	n_epochs = len(train_data) * kwargs['n_epochs']
	all_losses1, time_count1 = TriModel.train(
		dataset, model1, optimizer1, 1,
		n_epochs, kwargs['save_path'], kwargs['evaluate_every'],
		start_epoch1, autolr=False, autolr_every=200
	)
	all_losses2, time_count2 = TriModel.train(
		dataset, model2, optimizer2, 2,
		n_epochs, kwargs['save_path'], kwargs['evaluate_every'], 
		start_epoch2, autolr=False, autolr_every=200
	)
	all_losses3, time_count3 = TriModel.train(
		dataset, model3, optimizer3, 3,
		n_epochs, kwargs['save_path'], kwargs['evaluate_every'], 
		start_epoch3, autolr=False, autolr_every=200
	)

	# loss
	plt.figure(figsize=(15, 6))
	plt.plot(all_losses1, color='r', label='TextCNN')  
	plt.plot(all_losses2, color='g', label='BiGRU')  
	plt.plot(all_losses3, color='b', label='CNN-BiGRU')  
	plt.show()

	# time count
	fig = plt.figure(figsize=(15, 6))
	time_counts = [
		('TextCNN', time_count1),
		('BiGRU', time_count2),
		('CNNBiGRU', time_count3)
	]
	time_counts = sorted(time_counts, key=lambda p: p[1], reverse=True)		
	fig = sns.barplot(x=[name for name, _ in time_counts], y=[score for _, score in time_counts])
	fig.set(ylabel="time / min")
	fig.set(xlabel="models")
	fig.set(title="Time comparison")
	plt.show()

def trimodel_test(kwargs):
	'''
	args = {
		'n_epochs': 3,
		'batch_size': 1,
		'lr_rate': 0.001,
		'hidden_size': 64,
		'word2vec': False,
		'save_path': './runtime/evaluate_model',
		'evaluate_every': 300
	}
	'''
	train_data, test_data, train_label, test_label, lang, all_labels = load_data()
	input_size = lang.n_words
	output_size = len(all_labels)

	model1, start_epoch1 = load_previous_model(save_path = kwargs['save_path'] + '1')
	model2, start_epoch2 = load_previous_model(save_path = kwargs['save_path'] + '2')
	model3, start_epoch3 = load_previous_model(save_path = kwargs['save_path'] + '3')

	if 'word2vec' in kwargs and kwargs['word2vec'] is True:
		dataset = Dataset(train_data, train_label, lang, all_labels, batch_size=kwargs['batch_size'], word2vec=True)
		if model1 == None:
			model1 = TextCNN(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
				word_embedding=dataset.word_embedding, embed_size=dataset.embed_size
			)
		if model2 == None:
			model2 = BiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
				word_embedding=dataset.word_embedding, embed_size=dataset.embed_size
			)
		if model3 == None:
			model3 = CNNBiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], output_size = output_size, batch_size=kwargs['batch_size'], 
				word_embedding=dataset.word_embedding, embed_size=dataset.embed_size
			)
	else:
		dataset = Dataset(train_data, train_label, lang, all_labels, batch_size=kwargs['batch_size'])
		if model1 == None:
			model1 = TextCNN(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
			)
		if model2 == None:
			model2 = BiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				output_size = output_size, batch_size=kwargs['batch_size'], 
			)
		if model3 == None:
			model3 = CNNBiGRU(
				input_size=input_size, hidden_size=kwargs['hidden_size'], 
				kernel_num=256, kernel_sizes=[1], output_size = output_size, batch_size=kwargs['batch_size']
			)

	accuracy1 = TriModel.predict(dataset, model1, test_data, test_label, No=1)
	accuracy2 = TriModel.predict(dataset, model2, test_data, test_label, No=2)
	accuracy3 = TriModel.predict(dataset, model3, test_data, test_label, No=3)

	# save_accuracy
	if 'word2vec' in kwargs and kwargs['word2vec'] is True:
		save_accuracy({
			'TextCNN(word2vec)': accuracy1,
			'BiGRU(word2vec)': accuracy2,
			'CNNBiGRU(word2vec)': accuracy3
		})
	else:
		save_accuracy({
			'TextCNN': accuracy1,
			'BiGRU': accuracy2,
			'CNNBiGRU': accuracy3
		})

	accuracy = [
		('TextCNN', accuracy1),
		('BiGRU', accuracy2),
		('CNNBiGRU', accuracy3),
	]
	accuracy = sorted(accuracy, key=lambda p: p[1], reverse=True)
	plt.figure(figsize=(15, 6))
	fig = sns.barplot(x=[name for name, _ in accuracy], y=[score for _, score in accuracy])
	fig.set(ylabel="accuracy")
	fig.set(xlabel="models")
	fig.set(title="Accuracy comparison")
	plt.show()

	print('TextCNN accuracy', accuracy1)
	print('BiGRU accuracy', accuracy2)
	print('CNNBiGRU accuracy', accuracy3)

class TriModel(object):
	@staticmethod
	def train(dataset, model, optimizer, No, n_epochs, save_path, evaluate_every, start_epoch=1, autolr=False, autolr_every=200):
		model.train()
		runing_loss = 0
		all_losses = []
		start_time = time.time()

		if autolr:
			print('autolr')
			scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

		for epoch in range(start_epoch, n_epochs+1):
			optimizer.zero_grad()
			input_variable, target_variable, inputs_len = dataset.random_train()
			if No == 1:
				output_pred = model(input_variable)
			else:
				output_pred = model(input_variable, inputs_len, model.init_hidden())
			loss = F.cross_entropy(output_pred.view(target_variable.size(0), -1), target_variable)
			utils.clip_grad_norm(model.parameters(), max_norm=5)
			runing_loss += loss.data[0]
			#
			if autolr and epoch % autolr_every == 0 :
				scheduler.step(loss.data[0])

			if epoch % evaluate_every == 0:
				#print('lr_rate', optimizer.param_groups[0].get("lr"))
				save_model(model, epoch, save_path = save_path + str(No))
				k = 0
				topv, topi = output_pred.data.topk(1)
				pred = topi[0][0]
				out_index = target_variable.data.numpy()
				all_losses.append(runing_loss/evaluate_every)
				#print('loss = ', runing_loss/evaluate_every)
				runing_loss = 0

			loss.backward()
			optimizer.step()
		time_count = time.time() - start_time
		return all_losses, time_count/60

	@staticmethod
	def evaluate(dataset, model, data, No):
		#model.train(False)
		inp, inp_lens = dataset.evaluate_pad(data)
		input_tensor = torch.LongTensor(inp)
		input_variable = Variable(input_tensor, volatile=True).transpose(0, 1)
		if No == 1:
			output = model(input_variable)
		else:
			output = model(input_variable, inp_lens, None)
		top_v, top_i = output.data.topk(1)
		output_index = top_i[0][0]
		return dataset.all_labels[output_index], top_v[0][0], output_index
	@staticmethod
	def predict(dataset, model, test_data, test_label, No):
		total = 0
		correct = 0.0
		for idx in range(len(test_data)):
			data = test_data[idx]
			label = test_label[idx]
			if len(data) == 0:
				continue
			intent,top_value,top_index = TriModel.evaluate(dataset, model, data, No)
			if label==intent:
				correct += 1
			total += 1
		accuracy = correct/total
		return accuracy

def summary():
	dic = get_accuracy()
	accuracy = []
	for k,v in dic.items():
		accuracy.append((k, v))
	accuracy = sorted(accuracy, key=lambda p: p[1], reverse=True)
	plt.figure(figsize=(15, 6))
	fig = sns.barplot(x=[name for name, _ in accuracy], y=[score for _, score in accuracy])
	fig.set(ylabel="accuracy")
	fig.set(xlabel="models")
	fig.set(title="Accuracy comparison")
	plt.show()
