#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch
import torch.nn as nn
import time
import sys
from loguru import logger
import argparse
from tqdm import tqdm
from pathlib import Path
from data_handling.clotho_dataset import get_clotho_loader
from data_handling.audiocaps_dataset import get_audiocaps_loader
from tools.config_loader import get_config
from tools.utils import setup_seed, align_word_embedding, \
LabelSmoothingLoss, set_tgt_padding_mask, rotation_logger, \
decode_output, beam_search, greedy_decode, mixup_data
from tools.file_io import load_picke_file
from models.TransModel import TransformerModel
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler
from eval_metrics import evaluate_metrics


def train():

    start_time = time.time()

    batch_losses = torch.zeros(len(training_data))

    model.train()

    for batch_idx, train_batch in tqdm(enumerate(training_data), total=len(training_data)):

        src, tgt, f_names, tgt_len, captions = train_batch
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_pad_mask = set_tgt_padding_mask(tgt, tgt_len)

        optimizer.zero_grad()

        if config.training.mixup:
            tgt_a, tgt_b, lam, index = mixup_data(tgt, alpha=config.training.alpha)
            mixup_param = [lam, index]
            mixed_y_hat = model(src, tgt, mixup_param=mixup_param, target_padding_mask=tgt_pad_mask)

            tgt_a = tgt_a[:, 1:]
            tgt_b = tgt_b[:, 1:]
            mixed_y_hat = mixed_y_hat.transpose(0, 1)
            mixed_y_hat = mixed_y_hat[:, :tgt_a.size()[1], :]

            loss_a = criterion(mixed_y_hat.contiguous().view(-1, mixed_y_hat.size()[-1]),
                                tgt_a.contiguous().view(-1))
            loss_b = criterion(mixed_y_hat.contiguous().view(-1, mixed_y_hat.size()[-1]),
                                tgt_b.contiguous().view(-1))

            loss_mixup = lam * loss_a + (1 - lam) * loss_b

            loss_mixup.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

        y_hat = model(src, tgt, target_padding_mask=tgt_pad_mask)
        tgt = tgt[:, 1:]
        y_hat = y_hat.transpose(0, 1)  # batch x words_len x ntokens
        y_hat = y_hat[:, :tgt.size()[1], :]

        loss = criterion(y_hat.contiguous().view(-1, y_hat.size()[-1]),
                        tgt.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
        optimizer.step()

        if config.training.mixup:
            batch_losses[batch_idx] = loss.cpu().item() + loss_mixup.cpu().item()
        else:
            batch_losses[batch_idx] = loss.cpu().item()

    end_time = time.time()
    elasped_time = end_time - start_time
    epoch_loss = batch_losses.mean()
    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

    main_logger.info('epoch: {}, train_loss: {:.4f}, time elapsed: {:.4f}, lr:{:02.2e}'.format(epoch, epoch_loss, elasped_time, current_lr))


def eval_greedy(data, max_len=30):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_all = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            output = greedy_decode(model, src, sos_ind=sos_ind)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):    # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_all.extend(captions)
            file_names_all.extend(f_names)

        end_time = time.time()
        eval_time = end_time - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_all, file_names_all, words_list, log_output_dir)
        greedy_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = greedy_metrics['spider']['score']
        cider = greedy_metrics['cider']['score']
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'Spider score using greedy search: {spider:7.4f}, eval time: {eval_time:.4f}')


def eval_beam(data, beam_size, max_len=30):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_all = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            output = beam_search(model, src, sos_ind=sos_ind, eos_ind=eos_ind, beam_size=beam_size)

            y_hat_batch = torch.zeros([src.shape[0], max_len]).fill_(eos_ind).to(device)

            for i, o in enumerate(output):    # batch_size
                o = o[1:]
                for j in range(max_len - 1):
                    y_hat_batch[i, j] = o[j]
                    if o[j] == eos_ind:
                        break

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_all.extend(captions)
            file_names_all.extend(f_names)

        end_time = time.time()
        eval_time = end_time - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_all, file_names_all, words_list, log_output_dir, beam=True)
        beam_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = beam_metrics['spider']['score']
        cider = beam_metrics['cider']['score']
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')
        if config.mode != 'eval':
            spiders.append(spider)
            if spider >= max(spiders):
                torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": beam_size,
                        "epoch": epoch,
                        }, str(model_output_dir) + '/best_model.pt'.format(epoch))


parser = argparse.ArgumentParser(description='Settings for audio caption model')

parser.add_argument('-n', '--exp_name', type=str, default='exp1', help='name of the experiment')

args = parser.parse_args()

config = get_config()

setup_seed(config.training.seed)

exp_name = args.exp_name

# output setting
model_output_dir = Path('outputs', exp_name, 'model')
log_output_dir = Path('outputs', exp_name, 'logging')

model_output_dir.mkdir(parents=True, exist_ok=True)
log_output_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
            filter=lambda record: record['extra']['indent'] == 1)

logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
            filter=lambda record: record['extra']['indent'] == 1)

logger.add(str(log_output_dir) + '/captions.txt', format='{message}', level='INFO',
            filter=lambda record: record['extra']['indent'] == 2,
            rotation=rotation_logger)

logger.add(str(log_output_dir) + '/beam_captions.txt', format='{message}', level='INFO',
            filter=lambda record: record['extra']['indent'] == 3,
            rotation=rotation_logger)

main_logger = logger.bind(indent=1)

printer = PrettyPrinter()

device, device_name = (torch.device('cuda'), torch.cuda.get_device_name(torch.cuda.current_device()))

main_logger.info(f'Process on {device_name}')

dataset = config.data.type

batch_size = config.data.batch_size
num_workers = config.data.num_workers
input_field_name = config.data.input_field_name

# data loading
if dataset == 'clotho':
    words_list_path = config.path.clotho.words_list
    #words_freq_path = config.path.clotho.words_freq
    training_data = get_clotho_loader(split='development',
                                      input_field_name=input_field_name,
                                      load_into_memory=False,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=num_workers)

    validation_data = get_clotho_loader(split='validation',
                                        input_field_name=input_field_name,
                                        load_into_memory=False,
                                        batch_size=batch_size,
                                        num_workers=num_workers)

    evaluation_data = get_clotho_loader(split='evaluation',
                                    input_field_name=input_field_name,
                                    load_into_memory=False,
                                    batch_size=batch_size,
                                    num_workers=num_workers)
elif dataset == 'audiocaps':
    words_list_path = config.path.audiocaps.words_list
    #words_freq_path = config.path.audiocaps.words_freq
    training_data = get_audiocaps_loader(split='train',
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)

    evaluation_data = get_audiocaps_loader(split='test',
                                    batch_size=batch_size,
                                    num_workers=num_workers)

# loading vocabulary list
#words_list_path = 'data/pickles/new_words_list.p'
words_list = load_picke_file(words_list_path)
ntokens = len(words_list)
sos_ind = words_list.index('<sos>')
eos_ind = words_list.index('<eos>')

pretrained_cnn = torch.load(config.path.encoder + config.encoder.model + '.pth')['model'] if config.encoder.pretrained else None

pretrained_word_embedding = align_word_embedding(words_list, config.path.word2vec, config.decoder.nhid) if config.word_embedding.pretrained else None


main_logger.info('Training setting:\n'
            f'{printer.pformat(config)}')

model = TransformerModel(config, words_list, pretrained_cnn, pretrained_word_embedding)

model.to(device)

main_logger.info(f'Model:\n{model}\n')
main_logger.info('Total number of parameters:'
            f'{sum([i.numel() for i in model.parameters()])}')

main_logger.info(f'Len of training data: {len(training_data)}')
main_logger.info(f'Len of evaluation data: {len(evaluation_data)}')

if config.training.label_smoothing:
    criterion = LabelSmoothingLoss(ntokens, smoothing=0.1)
else:
    criterion = nn.CrossEntropyLoss()

spiders = []

if config.mode == 'train':

    main_logger.info('Training mode.')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    epochs = config.training.epochs
    ep = 1
    # warm up issue
    optimizer.zero_grad()
    optimizer.step()

    for epoch in range(ep, epochs + 1):

        scheduler_warmup.step(epoch)

        main_logger.info(f'Training epoch {epoch}...')
        train()
        main_logger.info('Evaluating...')
        eval_greedy(evaluation_data)
        eval_beam(evaluation_data, beam_size=2)
        eval_beam(evaluation_data, beam_size=3)
        eval_beam(evaluation_data, beam_size=4)
        eval_beam(evaluation_data, beam_size=5)
    main_logger.info('Training done.')

elif config.mode == 'finetune':

    main_logger.info('Finetune mode.')

    pretrained_model = torch.load(config.finetune.model)['model']
    if config.finetune.audiocap:
        dict_trained = pretrained_model
        dict_new = model.state_dict().copy()
        trained_list = [i for i in pretrained_model.keys()
                        if not (i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = dict_trained[trained_list[i]]
        model.load_state_dict(dict_new)
    else:
        model.load_state_dict(pretrained_model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.finetune.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    main_logger.info('Evaluating the baseline model performance.')
    eval_greedy(evaluation_data)
    epoch = 0
    eval_beam(evaluation_data, beam_size=2)
    eval_beam(evaluation_data, beam_size=3)
    eval_beam(evaluation_data, beam_size=4)
    eval_beam(evaluation_data, beam_size=5)

    optimizer.zero_grad()
    optimizer.step()

    epochs = config.finetune.epochs
    ep = 1

    for epoch in range(ep, epochs + 1):

        scheduler_warmup.step(epoch)

        main_logger.info(f'Finetune epoch {epoch}...')
        train()
        main_logger.info('Evaluating...')
        eval_greedy(evaluation_data)
        eval_beam(evaluation_data, beam_size=2)
        eval_beam(evaluation_data, beam_size=3)
        eval_beam(evaluation_data, beam_size=4)
        eval_beam(evaluation_data, beam_size=5)
    main_logger.info('Finetune done.')

elif config.mode == 'eval':

    main_logger.info('Evaluation mode.')

    model.load_state_dict(torch.load(config.path.model)['model'])
    main_logger.info('Metrcis on validation set')
    eval_greedy(validation_data)
    eval_beam(validation_data, beam_size=2)
    eval_beam(validation_data, beam_size=3)
    eval_beam(validation_data, beam_size=4)
    eval_beam(validation_data, beam_size=5)
    main_logger.info('Metrcis on evaluation set')
    eval_greedy(evaluation_data)
    eval_beam(evaluation_data, beam_size=2)
    eval_beam(evaluation_data, beam_size=3)
    eval_beam(evaluation_data, beam_size=4)
    eval_beam(evaluation_data, beam_size=5)
    main_logger.info('Evaluation done.')

