#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import platform
import numpy as np
import torch
import csv
import torch.nn as nn
import time
import sys
from loguru import logger
import argparse
from keywords import compute_keywords
from models.Tokenizer import WordTokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from data_handling.DataLoader import get_dataloader
from tools.config_loader import get_config
from tools.beam import beam_decode
from tools.utils import setup_seed, LabelSmoothingLoss, set_tgt_padding_mask, \
decode_output, greedy_decode, AverageMeter
from tools.file_io import load_pickle_file
from models.TransModel import TransformerModel
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler
from eval_metrics import evaluate_metrics


def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)

    # set up logger
    exp_name = config.exp_name

    if not config.encoder.pooling:
        folder_name = f'{exp_name}_data_{config.dataset}_seed_{config.training.seed}'
    else:
        folder_name = f'{exp_name}_data_{config.dataset}_pooling_type_{config.encoder.pooling_type}_seed_{config.training.seed}'

    # output setting
    model_output_dir = Path('outputs', folder_name, 'model')
    log_output_dir = Path('outputs', folder_name, 'logging')

    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
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

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    device, device_name = (torch.device('cuda'),
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())

    main_logger.info(f'Process on {device_name}')

    model = TransformerModel(config)
    model = model.to(device)

    if config.path.model != '':
        model.load_state_dict(torch.load(config.path.model)['model'])
        main_logger.info(f'Pre-trained model loaded from {config.path.model}')

    main_logger.info(f'Model:\n{model}\n')
    main_logger.info('Total number of parameters:'
                     f'{sum([i.numel() for i in model.parameters()])}')

    tokenizer = WordTokenizer(config)

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

    # get keywords for training set
    if config.keywords is True:
        num_keywords = config.num_keywords
        train_keywords_dict = load_pickle_file('data/Clotho/pickles/456/train_keywords_dict_pred_{}.p'.format(num_keywords))
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
            keywords_list[int(i/5)] = keywords_index

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

        test_keywords_dict = load_pickle_file('data/Clotho/pickles/456/test_keywords_dict_pred_{}.p'.format(num_keywords))
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

    # set up optimizer and loss
    if config.training.label_smoothing:
        criterion = LabelSmoothingLoss(ntokens, smoothing=0.1)
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    ep = 1

    # resume from a checkpoint
    if config.training.resume:
        checkpoint = torch.load(config.path.resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch']

    # training loop
    spiders = []
    for epoch in range(ep, config.training.epochs + 1):

        # training for one epoch
        main_logger.info(f'Training for epoch [{epoch}]')
        scheduler_warmup.step(epoch)

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            src, captions, audio_ids, _, f_names = batch_data
            src = src.to(device)
            tgt, tgt_len = tokenizer(captions)
            tgt = tgt.to(device)

            tgt_pad_mask = set_tgt_padding_mask(tgt, tgt_len)

            optimizer.zero_grad()

            if config.keywords:
                kw = torch.tensor(keywords_list[audio_ids.int().numpy()], dtype=torch.long)
                kw = kw.to(device)

            else:
                kw = None

            y_hat = model(src, tgt, keyword=kw, target_padding_mask=tgt_pad_mask)

            tgt = tgt[:, 1:]  # exclude <sos>
            y_hat = y_hat.transpose(0, 1)  # batch x words_len x ntokens
            y_hat = y_hat[:, :tgt.size()[1], :]  # truncate to the same length with target

            loss = criterion(y_hat.contiguous().view(-1, y_hat.size()[-1]),
                             tgt.contiguous().view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            epoch_loss.update(loss.cpu().item())

        elapsed_time = time.time() - start_time
        current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
        writer.add_scalar('train/loss', epoch_loss.avg, epoch)
        main_logger.info('epoch: {}, train_loss: {:.4f}, time elapsed: {:.4f}, lr:{:02.2e}'.
                         format(epoch, epoch_loss.avg, elapsed_time, current_lr))

        # validation loop, validation after each epoch
        main_logger.info("Validating...")

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
                        "config": config,
                    }, str(model_output_dir) + '/best_model.pth'.format(epoch))

    main_logger.info('Training done.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint in {best_epoch} th epoch.')
    main_logger.info(' Start evaluation.')
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
