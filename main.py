import time
import argparse
import pickle
from model import *
from utils import *
import os
import random
import numpy as np
from tqdm import tqdm
from os.path import join
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils1 import collate_fn
from dataset import load_data, RecSysDataset

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--dataset_path', default='yoochoose1_64/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--hiddenSize', type=int, default=128)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)


opt = parser.parse_args()
args=parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    init_seed(2020)
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)
    
    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn) 


    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = opt.dropout_gcn
        opt.dropout_global=opt.dropout_gcn
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = opt.dropout_gcn
        opt.dropout_global=opt.dropout_gcn
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = opt.dropout_gcn
        opt.dropout_global=opt.dropout_gcn
    elif opt.dataset == "RetailRocket":
        num_node = 36969
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = opt.dropout_gcn
        opt.dropout_global=opt.dropout_gcn
    elif opt.dataset == "yoochoose1_64":
        num_node = 37484
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = opt.dropout_gcn
        opt.dropout_global=opt.dropout_gcn
    else:
        num_node = 310

    train_data = pickle.load(open( opt.dataset + '/train.txt', 'rb'))
   # print(train_data)
   # print(preemb)
    opt.dropout_gcn = 0
    opt.dropout_local = opt.dropout_gcn
    opt.dropout_global=opt.dropout_gcn
    preemb=pickle.load(open(opt.dataset + '/'+opt.dataset+'_'+str(opt.hiddenSize)+'_'+'embedding.txt', 'rb'))
    preemb1=pickle.load(open(opt.dataset +'/'+opt.dataset+'_'+str(opt.hiddenSize)+'_'+'embedding1.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open( opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open( opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num,opt.hiddenSize,opt.hiddenSize,preemb.weight,preemb1.weight))
    #model = trans_to_cuda(CombineGraph(opt, num_node, adj, num,100,100))
  #  print(model.embedding.weight)
    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()