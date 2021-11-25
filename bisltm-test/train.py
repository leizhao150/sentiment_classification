# -*- coding: utf-8 -*-
# @Time    : 2021/5/26
# @Author  : leizhao150

import argparse
from model import train

# Get parameters from the command line
def init_args():
    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--att', '-att', default='True', type=str)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = init_args()
    is_add_att = False
    if args.att == 'True': is_add_att = True
    train(is_add_att)