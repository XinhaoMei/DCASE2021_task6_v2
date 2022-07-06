#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import platform
import sys
import argparse
import time
from pprint import PrettyPrinter

import torch
from sklearn.metrics import average_precision_score
from torch import nn
from tqdm import tqdm

from data_handling.TagLoader import get_dataloader
from models.Tag_model import Tag_model
from tools.config_loader import get_config
from tools.file_io import load_pickle_file, write_pickle_file
from tools.utils import setup_seed, AverageMeter
from pathlib import Path
from loguru import logger


def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)

    # set up logger
    exp_name = config.exp_name

    # output setting
    model_output_dir = Path('outputs', 'tagging', exp_name, 'model')
    log_output_dir = Path('outputs', 'tagging', exp_name, 'logging')

    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    config.wav.sr = 44100
    config.wav.window_size = 1024
    config.wav.hop_length = 512

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    device, device_name = (torch.device('cuda'),
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())

    main_logger.info(f'Process on {device_name}')

    model = Tag_model(config)
    model = model.to(device)

    main_logger.info(f'Model:\n{model}\n')
    main_logger.info('Total number of parameters:'
                     f'{sum([i.numel() for i in model.parameters()])}')

    tag_vocab = load_pickle_file('data/Clotho/pickles/456/tag_vocab_456.p')
    main_logger.info(f'Total {len(tag_vocab)} audio classes.')

    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    test_loader = get_dataloader('test', config)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}; Number of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}; Number of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}; Number of batches: {len(test_loader)}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    epochs = 30
    val_losses = []

    for ep in range(1, epochs + 1):
        main_logger.info(f'Training {ep}th epoch')

        start_time = time.time()
        model.train()

        epoch_loss = AverageMeter()
        epoch_precisions = AverageMeter()
        for batch_idx, train_batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y, _, _ = train_batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)  # .sum(dim=1).mean(dim=0)

            loss.backward()
            optimizer.step()

            running_prec = average_precision_score(y.cpu().view(-1),
                                                   torch.sigmoid(y_hat).detach().cpu().view(-1))

            epoch_loss.update(loss.cpu().item())
            epoch_precisions.update(running_prec)

        main_logger.info('Epoch: {}, train_loss:{:.4f}, precision: {:.4f}, time:{:.4f}'.
                         format(ep, epoch_loss.avg, epoch_precisions.avg, time.time() - start_time))

        main_logger.info('Validating...')
        start_time = time.time()
        val_loss, val_precision = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        main_logger.info('Epoch: {}, val_loss:{:.4f}, val_precision: {:.4f}, time:{:.4f}'.
                         format(ep, val_loss, val_precision, time.time() - start_time))
        if val_loss <= min(val_losses):
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': ep,
            }, str(model_output_dir) + '/best_model.pth')

    main_logger.info('Training done. Start evaluation')
    start_time = time.time()
    checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(checkpoint['model'])
    best_epoch = checkpoint['epoch']
    main_logger.info(f'Best checkpoint in {best_epoch} th epoch.')
    test_loss, test_precision = validate(model, test_loader, criterion, device)
    main_logger.info('Test_loss:{:.4f}, test_precision: {:.4f}, time:{:.4f}'.
                     format(test_loss, test_precision, time.time() - start_time))


def validate(model, data_loader, criterion, device):

    model.eval()

    epoch_loss = AverageMeter()
    epoch_precisions = AverageMeter()
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            x, y, _, _ = batch_data
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)  # .sum(dim=1).mean(dim=0)

            epoch_loss.update(loss.cpu().item())
            running_prec = average_precision_score(y.cpu().view(-1),
                                                   torch.sigmoid(y_hat).detach().cpu().view(-1))

            epoch_precisions.update(running_prec)

    return epoch_loss.avg, epoch_precisions.avg


def predict(config):

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    config.wav.sr = 44100
    config.wav.window_size = 1024
    config.wav.hop_length = 512

    # set up model
    device, device_name = (torch.device('cuda'),
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())

    main_logger.info(f'Process on {device_name}')

    tag_vocab = load_pickle_file('data/Clotho/pickles/456/tag_vocab_456.p')

    model = Tag_model(config)
    model = model.to(device)

    model.load_state_dict(torch.load('outputs/tagging/tag_456/model/best_model.pth')['model'])

    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    test_loader = get_dataloader('test', config)

    train_prediction, val_prediction, test_prediction = {}, {}, {}

    audio_names = []
    indexs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y, _, audio_name = batch_data
            x = x.to(device)

            y_hat = torch.sigmoid(model(x))

            _, kw_index = torch.topk(y_hat, 5, dim=-1)

            audio_names.extend(audio_name)
            indexs.extend(kw_index)
        for name, index in zip(audio_names, indexs):
            kws = [tag_vocab[i] for i in index]
            train_prediction[name] = kws

    audio_names = []
    indexs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y, _, audio_name = batch_data
            x = x.to(device)

            y_hat = torch.sigmoid(model(x))

            _, kw_index = torch.topk(y_hat, 5, dim=-1)

            audio_names.extend(audio_name)
            indexs.extend(kw_index)
        for name, index in zip(audio_names, indexs):
            kws = [tag_vocab[i] for i in index]
            val_prediction[name] = kws

    audio_names = []
    indexs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, y, _, audio_name = batch_data
            x = x.to(device)

            y_hat = torch.sigmoid(model(x))

            _, kw_index = torch.topk(y_hat, 5, dim=-1)

            audio_names.extend(audio_name)
            indexs.extend(kw_index)
        for name, index in zip(audio_names, indexs):
            kws = [tag_vocab[i] for i in index]
            test_prediction[name] = kws

    write_pickle_file(val_prediction, 'data/Clotho/pickles/456/train_keywords_dict_pred.p')
    write_pickle_file(val_prediction, 'data/Clotho/pickles/456/val_keywords_dict_pred.p')
    write_pickle_file(test_prediction, 'data/Clotho/pickles/456/test_keywords_dict_pred.p')


if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-e', '--batch', default=16, type=int,
                        help='Batch size.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed')

    args = parser.parse_args()

    config = get_config(args.config)

    config.exp_name = args.exp_name
    config.data.batch_size = args.batch
    config.training.seed = args.seed
    train(config)
    # predict(config)
