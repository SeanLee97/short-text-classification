# !/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os, codecs
import jieba
import numpy as np

class Dataset(object):  
        def __init__(self, corpus='./corpus/data'):
                self.corpus = corpus
        
        # 遍历目录
        def traveldir(self, path, files=[]):
                for filename in os.listdir(path):
                        if '.DS_Store' in filename:
                                continue
                        filepath = os.path.join(path, filename)
                        if os.path.isdir(filepath):
                                self.traveldir(filepath, files)
                        else:
                                files.append(filepath)
                return files

        def get_files(self):
                files = self.traveldir(self.corpus)
                return files
        
        # 加载数据, yield line, label(path作为label)
        def load_data_label(self):
                files = self.get_files()
                for path in files:
                        with open(path, 'r') as f:
                                lines = f.readlines()
                                if lines[:3] == codecs.BOM_UTF8:
                                        lines = lines[3:]
                                for line in lines:
                                        yield line.strip(), path
        def load_data(self):
                files = self.get_files()
                for path in files:
                        with open(path, 'r') as f:
                                lines = f.readlines()
                                if lines[:3] == codecs.BOM_UTF8:
                                        lines = lines[3:]
                                for line in lines:
                                        yield line.strip()
class DatasetIter(object):
        def __iter__(self):
                for line in Dataset().load_data():
                        yield [x for x in jieba.lcut(line)]


class Word2vec(object):
        def __init__(self, savedir='./runtime/word2vec.bin'):
                self.savedir = savedir

        def train(self):
                data = DatasetIter()
                self.model = Word2Vec(data, min_count=1)
                self.model.wv.save_word2vec_format(self.savedir, binary=True)

        def load(self):
                if not os.path.exists(self.savedir):
                        self.train()
                self.model = KeyedVectors.load_word2vec_format(self.savedir, binary=True)
                return self.model

        def word_vector(self, sentence):
                word2vec_numpy = list()
                for word in sentence:
                        if word in self.model:
                                word2vec_numpy.append(self.model[word].tolist())
                embed_size = len(word2vec_numpy[0])
                col = []
                for i in range(embed_size):
                        count = 0.0
                        for j in range(int(len(word2vec_numpy))):
                                count += word2vec_numpy[j][i]
                                count = round(count, 6)
                        col.append(count)
                zero = []
                for m in range(embed_size):
                        avg = col[m] / (len(word2vec_numpy))
                        avg = round(avg, 6)
                        zero.append(float(avg))
                list_word2vec = []
                oov = 0
                iov = 0
                for word in sentence:
                        if word not in self.model:
                                oov += 1
                                word2vec = zero
                        else:
                                iov += 1
                                word2vec = self.model[word].tolist()
                        list_word2vec.append(word2vec)
                embed_size = len(list_word2vec[0])
                return list_word2vec, embed_size
if __name__=='__main__':
        word2vec = Word2vec()
        #word2vec.train()
        model = word2vec.load()
        if '哈哈' in model:
                print(model.most_similar('哈哈'))
