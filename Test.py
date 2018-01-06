# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class Config(object):
    def __init__(self, data_path,dimension,num_entity,num_relation):
        self.data_path = data_path
        self.dimension = dimension
        self.num_entity = num_entity
        self.num_relation = num_relation

class TripleData(Dataset):
    def __init__(self, file_path,test,flag):
        with open(file_path, 'r') as f:
            data = []
            for line in f.readlines():
                triplet = [int(s) for s in line.split('\t')]
                test.add(tuple(triplet),flag)   # add tuple
                data.append(triplet)
        self.data = data
        self.data_set = set(tuple(i) for i in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class TripleLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super(TripleLoader, self).__init__(dataset=dataset, batch_size=batch_size)

def is_empty(s):
    return len(s) != 1

def tensor2numpy0(tensor):
    return tensor.numpy()[0]

def valueAdd(dict,key,value):
    if key in dict:
        dict[key] += value
    else:
        dict[key] = value

class Test():
    def __init__(self,config):
        self.ok = {}
        self.fb_h = []
        self.fb_r = []
        self.fb_t = []
        self.rel_num = {}
        self.dim_embedding = config.dimension
        self.embed_entity = nn.Embedding(config.num_entity, config.dimension)
        self.embed_relation = nn.Embedding(config.num_relation, config.dimension)
        self.config = config

    def add(self,triplet,flag):
        # print(triplet)
        if flag:
            self.fb_h.append(triplet[0])
            self.fb_r.append(triplet[1])
            self.fb_t.append(triplet[2])
            self.rel_num[triplet[1]] = 0
        self.ok[triplet] = 1

    def loadTestData(self):
        self.test_data = TripleData(os.path.join('data/fb15k/encode/test_encode.txt'),self,True)
        self.test_loader = TripleLoader(self.test_data,1)

    def loadTrainingData(self):
        self.train_data = TripleData(os.path.join('data/fb15k/encode/train_encode.txt'),self,False)

    def loadValidData(self):
        self.valid_data = TripleData(os.path.join('data/fb15k/encode/valid_encode.txt'),self,False)

    def loadEmbedding(self):
        with open(os.path.join(self.config.data_path, 'encode/entity2vec.bern')) as f:
            self.embed_entity = Variable(torch.FloatTensor([[float(s) for s in list(filter(is_empty,line.split('\t')))] for line in f.readlines()]))
        with open(os.path.join(self.config.data_path,'encode/relation2vec.bern')) as f:
            self.embed_relation = Variable(torch.FloatTensor([[float(s) for s in list(filter(is_empty,line.split('\t')))] for line in f.readlines()]))

    def run(self):
        lsum, lsum_filter = 0, 0
        rsum, rsum_filter = 0, 0
        lp_n, lp_n_filter = 0, 0
        rp_n, rp_n_filter = 0, 0

        lsum_r, lsum_filter_r = {}, {}
        rsum_r, rsum_filter_r = {}, {}
        lp_n_r, lp_n_filter_r = {}, {}
        rp_n_r, rp_n_filter_r = {}, {}

        for step, data_batch in enumerate(self.test_loader, 0):
            print('step', step)
            h,r,t = data_batch[0],data_batch[1],data_batch[2]
            _rate = torch.norm(self.embed_entity[h] + self.embed_relation[r] - self.embed_entity[t], p=2, dim=1, keepdim=True)
            # print('rate',_rate)
            self.rel_num[tensor2numpy0(r)] += 1

            # substitute head entity
            # print("substitute head entity")
            rank = {}
            for index in range(self.config.num_entity):
                rate = torch.norm(self.embed_entity[index] + self.embed_relation[r] - self.embed_entity[t], p=2, dim=1, keepdim=True)
                rank[index] = rate
            rank = sorted(rank.items(),key=lambda d:d[1].data.numpy()[0][0],reverse = False)
            # print(rank)

            filter = 0
            for i in range(len(rank)):
                if not (rank[i][0],tensor2numpy0(r),tensor2numpy0(t)) in self.ok:
                    filter += 1

                # correct triplet
                if rank[i][0] == tensor2numpy0(h):
                    # print("rank",i)
                    # print("filter rank",filter)
                    lsum += i
                    lsum_filter += filter
                    valueAdd(lsum_r, r, i)
                    valueAdd(lsum_filter_r, r, filter)
                    if i <= 10:
                        lp_n += 1
                        valueAdd(lp_n_r, r, 1)
                    if filter <= 10:
                        lp_n_filter += 1
                        valueAdd(lp_n_filter_r, r, 1)
                    break
            # print("===========================================")

            # substitute tail entity
            # print("substitute tail entity")
            rank = {}
            for index in range(self.config.num_entity):
                rate = torch.norm(self.embed_entity[h] + self.embed_relation[r] - self.embed_entity[index], p=2, dim=1, keepdim=True)
                rank[index] = rate
            rank = sorted(rank.items(),key=lambda d:d[1].data.numpy()[0][0],reverse = False)
            # print(rank)

            filter = 0
            for i in range(len(rank)):
                if not (tensor2numpy0(h),tensor2numpy0(r),rank[i][0]) in self.ok:
                    filter += 1

                # correct triplet
                if rank[i][0] == tensor2numpy0(t):
                    # print("rank",i)
                    # print("filter rank",filter)
                    rsum += i
                    rsum_filter += filter
                    valueAdd(rsum_r, r, i)
                    valueAdd(rsum_filter_r, r, filter)
                    if i <= 10:
                        rp_n += 1
                        valueAdd(rp_n_r, r, 1)
                    if filter <= 10:
                        rp_n_filter += 1
                        valueAdd(rp_n_filter_r, r, 1)
                    break
            # print("===========================================")
        print("left",lsum/len(self.fb_t),lp_n/len(self.fb_t),lsum_filter/len(self.fb_t),lp_n_filter/len(self.fb_t))
        print("right",rsum/len(self.fb_r),rp_n/len(self.fb_t), rsum_filter/len(self.fb_t),rp_n_filter/len(self.fb_t))

def main():
    config = Config(data_path='data/fb15k/',dimension=100,num_entity = 0,num_relation = 0)
    with open(os.path.join(config.data_path, 'encode/entity_id.txt')) as f:
        entity2id = {line.split('\t')[0].strip() : line.split('\t')[1].strip() for line in f.readlines()}
    with open(os.path.join(config.data_path, 'encode/relation_id.txt')) as f:
        relation2id = {line.split('\t')[0].strip() : line.split('\t')[1].strip() for line in f.readlines()}
    id2entity = {value : key for key, value in entity2id.items()}
    id2relation = {value : key for key, value in relation2id.items()}
    config.num_entity = len(entity2id)
    config.num_relation = len(relation2id)

    test = Test(config)

    # load test data
    print('Reading test data...')
    test.loadTestData()

    # load traning data
    print('Reading training data...')
    test.loadTrainingData()

    # load valid data
    print('Reading valid data...')
    test.loadValidData()

    # load embeddings
    print('loading embeddings...')
    test.loadEmbedding()

    # run
    test.run()

if __name__ == '__main__':
    main()