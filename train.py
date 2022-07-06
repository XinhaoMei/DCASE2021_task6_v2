#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
from trainer.trainer import train
from tools.config_loader import get_config


if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-d', '--dataset', default='Clotho', type=str,
                        help='Dataset used.')
    parser.add_argument('-w', '--word', default='True', type=str,
                        help='Pre-trained word embedding.')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-e', '--batch', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed')
    parser.add_argument('-k', '--keywords', default='True', type=str,
                        help='Use keywords or not.')
    parser.add_argument('-p', '--pooling', default='True', type=str,
                        help='Use input pooling or not.')
    parser.add_argument('-t', '--type', default='avg', type=str,
                        help='Input pooling type')

    args = parser.parse_args()

    config = get_config(args.config)

    config.exp_name = args.exp_name
    config.dataset = args.dataset
    config.data.batch_size = args.batch
    config.training.seed = args.seed
    config.word_embedding.pretrained = eval(args.word)
    config.keywords = eval(args.keywords)
    config.encoder.pooling = eval(args.pooling)
    config.encoder.pooling_type = args.type
    train(config)
