# -*- coding: utf-8 -*-
# @Time    : 2021/5/26
# @Author  : leizhao150
import argparse
from model import train

# Get parameters from the command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using BILSTM-CRF.')
    # Input/output options
    parser.add_argument('--pr', '-pr', default='bert', type=str)
    parser.add_argument('--bs', '-bs', default=16, type=int)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = init_args()
    train(args.pr, args.bs)