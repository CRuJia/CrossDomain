import numpy as np
import os
import argparse

def parse_args(script):
    parser = argparse.ArgumentParser(description="few-shot script %s"%(script))
    parser.add_argument('--trainset', default=['miniImagenet','cars'], type=list,help="miniImagenet/cub/cars/places")
    parser.add_argument("--testset",default='cub', type=str, help="cub/cars/places, valid only when dataset=multi")
    parser.add_argument("--model", default="Conv4", help="Conv{4|6}/ ResNet{10|18|34")
    parser.add_argument("--method", default="relationnet", help="baseline/baseline++/protonet/relationnet")
    parser.add_argument("--train_n_way", default=5, type=int, help="class num to classify for training")
    parser.add_argument("--test_n_way", default=5, type=int, help="class num to classify for testing(validation)")
    parser.add_argument("--n_support", default=5, type=int, help="number of labeled data in each class in support set")
    parser.add_argument("--n_query", default=5, type=int, help="number of labeled data in each class in query set")
    parser.add_argument("--n_episode", default=100, type=int, help="") #TODO
    parser.add_argument('--name', default='tmp', type=str, help='')
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument("--use_cuda", default=True, type=bool, help="use cuda or not")
    parser.add_argument('--save_dir', default='./output', type=str, help='')

    if script == 'train':
        # parser.add_argument('--num_classes', default=200, type=int, help="total number of classes in softmax, only used in baseline")
        parser.add_argument('--save_freq', default=25, type=int, help='Save frequency')
        parser.add_argument("--start_epoch", default=0, type=int, help="starting epoch")
        parser.add_argument("--end_epoch", default=50, type=int, help="stoping epoch")
        parser.add_argument("--resume", default="",type=str, help="continuous from previous trained model with largest epoch")
        parser.add_argument("--reseme_epoch", default=1, type=int, help="")

    return parser.parse_args()
