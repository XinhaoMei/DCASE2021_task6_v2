#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import os
import sys
import torch
import platform
import torch.nn as nn
import numpy as np
import time
from loguru import logger
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from tools.beam import beam_decode
from tools.config_loader import get_config
from data_handling.DataLoader import get_dataloader
from tools.utils import setup_seed, decode_output, greedy_decode, AverageMeter
from models.Tokenizer import WordTokenizer
from tools.file_io import load_pickle_file
from models.TransModel import TransformerModel
from pprint import PrettyPrinter
from tools.rl_utils import scst_sample, get_self_critical_reward, pack_sample
from eval_metrics import evaluate_metrics
from warmup_scheduler import GradualWarmupScheduler


def train(config):

    setup_seed(config.training.seed)

    exp_name = config.exp_name

    folder_name = f'{exp_name}_data_{config.dataset}_seed_{config.training.seed}'

    model_output_dir = Path('rl_outputs', folder_name, 'model')
    log_output_dir = Path('rl_outputs', folder_name, 'logging')

    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}',
               level='INFO', filter=lambda record: record['extra']['indent'] == 1)

    logger.add(log_output_dir.joinpath('output.txt'),
               format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_output_dir) + '/tensorboard')

    # wav settings
    if config.dataset == 'AudioCaps':
        config.wav.sr = 32000
        config.wav.window_size = 1024
        config.wav.hop_length = 320
    elif config.dataset == 'Clotho':
        config.wav.sr = 44100
        config.wav.window_size = 1024
        config.wav.hop_length = 512

    printer = PrettyPrinter()

    device, device_name = (torch.device('cuda'), torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())

    main_logger.info('Finetune using Reinforcement Learning.')
    main_logger.info(f'Process on {device_name}')

    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config, return_dict=True)
    test_loader = get_dataloader('test', config, return_dict=True)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}; Number of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}; Number of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}; Number of batches: {len(test_loader)}')

    vocabulary = load_pickle_file(config.path.vocabulary.format(config.dataset))
    ntokens = len(vocabulary)
    sos_ind = vocabulary.index('<sos>')
    eos_ind = vocabulary.index('<eos>')

    checkpoint = torch.load(config.rl.model)
    main_logger.info(f'Loading cp from {config.rl.model}')
    config = checkpoint['config']

    model = TransformerModel(config)
    model = model.to(device)

    # get keywords for training set
    if config.keywords is True:
        num_keywords = config.num_keywords
        train_keywords_dict = load_pickle_file(
            'data/Clotho/pickles/456/train_keywords_dict_pred_{}.p'.format(num_keywords))
        train_size = len(train_loader.dataset)
        keywords_list = np.zeros((int(train_size / 5), num_keywords))
        for i in range(0, train_size, 5):
            file_name = train_loader.dataset[i][-1]
            keywords = train_keywords_dict[file_name]
            keywords_index = [vocabulary.index(word) for word in keywords]
            while len(keywords_index) < num_keywords:
                keywords_index.append(keywords_index[-1])
            if len(keywords_index) > num_keywords:
                keywords_index = keywords_index[:num_keywords]
            keywords_list[int(i / 5)] = keywords_index

        # val_keywords, test_keywords = compute_keywords(config, train_loader, val_loader, test_loader, keywords_list)
        val_keywords_dict = load_pickle_file('data/Clotho/pickles/456/val_keywords_dict_pred_{}.p'.format(num_keywords))
        val_size = len(val_loader.dataset)
        val_keywords = np.zeros((val_size, num_keywords))
        for i in range(val_size):
            file_name = val_loader.dataset[i][-1]
            keywords = val_keywords_dict[file_name]
            keywords_index = [vocabulary.index(word) for word in keywords]
            while len(keywords_index) < num_keywords:
                keywords_index.append(keywords_index[-1])
            val_keywords[i] = keywords_index

        test_keywords_dict = load_pickle_file(
            'data/Clotho/pickles/456/test_keywords_dict_pred_{}.p'.format(num_keywords))
        test_size = len(test_loader.dataset)
        test_keywords = np.zeros((test_size, num_keywords))
        for i in range(test_size):
            file_name = test_loader.dataset[i][-1]
            keywords = test_keywords_dict[file_name]
            keywords_index = [vocabulary.index(word) for word in keywords]
            while len(keywords_index) < num_keywords:
                keywords_index.append(keywords_index[-1])
            test_keywords[i] = keywords_index
    else:
        val_keywords, test_keywords = None, None

    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5, weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

    if not model_output_dir.joinpath('best_model.pth').exists():
        main_logger.info('Training settings:\n'
                         f'{printer.pformat(config)}')
        main_logger.info(f'Model:\n{model}\n')
        main_logger.info('Total number of parameters:'
                         f'{sum([i.numel() for i in model.parameters()])}')
        ep = 1
        model.load_state_dict(checkpoint['model'])
        main_logger.info('Evaluating baseline performance...')
        for i in range(1, 4):
            validate(test_loader, model, beam_size=i, sos_ind=sos_ind,
                     eos_ind=eos_ind, vocabulary=vocabulary, log_dir=log_output_dir,
                     epoch=0, device=device, is_keyword=config.keywords, keywords_list=test_keywords)
        spiders = []
    else:
        main_logger.info('Resume from last training checkpoint')
        checkpoint = torch.load(str(model_output_dir)+'/best_model.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch'] + 1
        spiders = checkpoint['spiders']

    for name, param in model.feature_extractor.named_parameters():
        param.requires_grad = False

    epochs = 100

    for epoch in range(ep, epochs + 1):

        main_logger.info(f'Finetune using RL epoch {epoch}...')

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            src, captions, audio_ids, _, f_names = batch_data
            src = src.to(device)

            if config.keywords:
                kw = torch.tensor(keywords_list[audio_ids.int().numpy()], dtype=torch.long)
                kw = kw.to(device)
            else:
                kw = None

            model.eval()
            with torch.no_grad():
                    output = greedy_decode(model, src, keyword=kw, sos_ind=sos_ind, eos_ind=eos_ind)
                    output = output[:, 1:].int()
                    out_tensor = torch.zeros(output.shape).fill_(eos_ind).to(device)
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            out_tensor[i, j] = output[i, j]
                            if output[i, j] == eos_ind:
                                break
                            elif j == output.shape[1] - 1:
                                out_tensor[i, j] = eos_ind

            model.train()
            sample, sampled_logprobs = scst_sample(model, src, keyword=kw, sos_ind=sos_ind, eos_ind=eos_ind)
            sample, _ = pack_sample(sample, sos_ind, eos_ind)

            optimizer.zero_grad()
            reward_score, sample_score = get_self_critical_reward(out_tensor, sample, captions, vocabulary, sos_ind, eos_ind)
            reward = np.repeat(reward_score[:, np.newaxis], sample.size(-1), 1)
            reward = torch.as_tensor(reward).float().to(device)
            mask = (sample != eos_ind).float()
            mask = torch.cat([torch.ones(mask.size(0), 1).to(device), mask[:, :-1]], 1).float()
            loss = - sampled_logprobs * reward * mask
            loss = loss.to(device)
            loss = torch.sum(loss, dim=1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            epoch_loss.update(loss.cpu().item())

        elasped_time = time.time() - start_time
        ep_loss = epoch_loss.avg
        current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
        main_logger.info('epoch: {}, train_loss: {:.4f}, '
                         'time elapsed: {:.4f}, lr:{:02.2e}'.format(epoch,
                                                                    ep_loss,
                                                                    elasped_time,
                                                                    current_lr))

        main_logger.info('Validating...')
        for i in range(1, 4):
            spider = validate(val_loader, model, beam_size=i, sos_ind=sos_ind,
                              eos_ind=eos_ind, vocabulary=vocabulary, log_dir=log_output_dir,
                              epoch=epoch, device=device, is_keyword=config.keywords,
                              keywords_list=val_keywords)['spider']['score']
            if i != 1:
                spiders.append(spider)
                if spider >= max(spiders):
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": i,
                        "epoch": epoch,
                        "spiders": spiders,
                    }, str(model_output_dir) + '/best_model.pth'.format(epoch))
                    main_logger.info('Model saved!')
    main_logger.info('Finetune done.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint in {best_epoch} th epoch.')
    main_logger.info('Start evaluation.')
    for i in range(1, 4):
        validate(test_loader, model, beam_size=i, sos_ind=sos_ind,
                 eos_ind=eos_ind, vocabulary=vocabulary, log_dir=log_output_dir,
                 epoch=0, device=device, is_keyword=config.keywords, keywords_list=test_keywords)
    main_logger.info('Evaluation done.')


def validate(data_loader, model, beam_size, sos_ind, eos_ind, vocabulary, log_dir, epoch, device, is_keyword, keywords_list):

    val_logger = logger.bind(indent=1)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data_loader), total=len(data_loader)):

            src, target_dicts, audio_ids, _, file_names = eval_batch
            src = src.to(device)

            if is_keyword:
                kw = torch.tensor(keywords_list[audio_ids.int().numpy()], dtype=torch.long)
                kw = kw.to(device)
            else:
                kw = None

            if beam_size == 1:
                output = greedy_decode(model, src, keyword=kw, sos_ind=sos_ind, eos_ind=eos_ind)
            else:
                output = beam_decode(src, model, keyword=kw, sos_ind=sos_ind, eos_ind=eos_ind, beam_width=beam_size)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):    # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break
                    elif j == output.shape[1] - 1:
                        y_hat_batch[i, j] = eos_ind

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_dict.extend(target_dicts)
            file_names_all.extend(file_names)

        eval_time = time.time() - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all,
                                                   vocabulary, log_dir, epoch, beam_size=beam_size)
        metrics = evaluate_metrics(captions_pred, captions_gt)

        spider = metrics['spider']['score']
        cider = metrics['cider']['score']

        val_logger.info(f'Cider: {cider:7.4f}')
        val_logger.info(
            f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')

        if beam_size == 3 and (epoch % 5) == 0:
            for metric, values in metrics.items():
                val_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')

        return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune using Reinforcement learning')

    parser.add_argument('-n', '--exp_name', type=str, default='exp1', help='name of the experiment')
    parser.add_argument('-d', '--dataset', default='Clotho', type=str,
                        help='Dataset used.')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-e', '--batch', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed')

    args = parser.parse_args()

    config = get_config(args.config)

    config.exp_name = args.exp_name
    config.dataset = args.dataset
    config.data.batch_size = args.batch
    config.training.seed = args.seed
    train(config)
