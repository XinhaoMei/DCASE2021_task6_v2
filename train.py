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
from tools.config_loader import get_config
from tools.utils import setup_seed, align_word_embedding, \
LabelSmoothingLoss, set_tgt_padding_mask, rotation_logger, \
decode_output, beam_search, greedy_decode, mixup_data
from tools.file_io import load_picke_file
from models.TransModel import TransformerModel
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler
from eval_metrics import evaluate_metrics


setup_seed(20)

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
            mixed_src, tgt_a, tgt_b, lam, index = mixup_data(src, tgt, alpha=config.training.alpha)
            mixup_param = [lam, index]
            mixed_y_hat = model(mixed_src, tgt, mixup_param=mixup_param, target_padding_mask=tgt_pad_mask)

            tgt_a = tgt_a[:, 1:]
            tgt_b = tgt_b[:, 1:]
            mixed_y_hat = mixed_y_hat.transpose(0, 1)
            mixed_y_hat = mixed_y_hat[:, :tgt_a.size()[1], :]

            loss_a = criterion(mixed_y_hat.contiguous().view(-1, mixed_y_hat.size()[-1]),
                                tgt_a.contiguous().view(-1))
            loss_b = criterion(mixed_y_hat.contiguous().view(-1, mixed_y_hat.size()[-1]),
                                tgt_b.contiguous().view(-1))
            loss = lam * loss_a + (1 - lam) * loss_b
        else:

            y_hat = model(src, tgt, target_padding_mask=tgt_pad_mask)
            tgt = tgt[:, 1:]
            y_hat = y_hat.transpose(0, 1)  # batch x words_len x ntokens
            y_hat = y_hat[:, :tgt.size()[1], :]

            loss = criterion(y_hat.contiguous().view(-1, y_hat.size()[-1]), 
                            tgt.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
        optimizer.step()

        batch_losses[batch_idx] = loss.cpu().item()

    end_time = time.time()
    elasped_time = end_time - start_time
    epoch_loss = batch_losses.mean()
    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

    main_logger.info('epoch: {}, train_loss: {:.4f}, time elapsed: {:.4f}, lr:{:02.2e}'.format(epoch, epoch_loss, elasped_time, current_lr))


def eval_greedy(data, sos_ind=0, eos_ind=9, max_len=30):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_all = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            output = greedy_decode(model, src)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):    # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.detach().cpu())
            ref_captions_all.extend(captions)
            file_names_all.extend(f_names)

        end_time = time.time()
        eval_time = end_time - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_all, file_names_all, words_list, log_output_dir)
        greedy_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = greedy_metrics['spider']['score']
        main_logger.info(f'Spider score using greedy search: {spider:7.4f}, eval time: {eval_time:.4f}')


def eval_beam(data, beam_size, sos_ind=0, eos_ind=9, max_len=30):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_all = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            output = beam_search(model, src, beam_size=beam_size)

            y_hat_batch = torch.zeros([src.shape[0], max_len]).fill_(eos_ind).to(device)

            for i, o in enumerate(output):    # batch_size
                o = o[1:]
                for j in range(max_len - 1):
                    y_hat_batch[i, j] = o[j]
                    if o[j] == eos_ind:
                        break

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.detach().cpu())
            ref_captions_all.extend(captions)
            file_names_all.extend(f_names)

        end_time = time.time()
        eval_time = end_time - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_all, file_names_all, words_list, log_output_dir, beam=True)
        beam_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = beam_metrics['spider']['score']
        main_logger.info(f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')


parser = argparse.ArgumentParser(description='Settings for audio caption model')

parser.add_argument('-n', '--exp_name', type=str, default='exp1', help='name of the experiment')

args = parser.parse_args()

config = get_config()

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

words_list_path = config.path.words_list
words_freq_path = config.path.words_freq

# loading vocabulary list
words_list = load_picke_file(words_list_path)
ntokens = len(words_list)


pretrained_cnn = torch.load(config.path.encoder + config.encoder.model + '.pth')['model'] if config.encoder.pretrained else None

pretrained_word_embedding = align_word_embedding(words_list, config.path.word2vec, config.decoder.nhid) if config.word_embedding.pretrained else None


main_logger.info('Training setting:\n'
            f'{printer.pformat(config)}')

model = TransformerModel(config, words_list, pretrained_cnn, pretrained_word_embedding)

model.to(device)

main_logger.info(f'Model:\n{model}\n')
main_logger.info('Total number of parameters:'
            f'{sum([i.numel() for i in model.parameters()])}')

batch_size = config.data.batch_size
num_workers = config.data.num_workers
input_field_name = config.data.input_field_name

# data loading
training_data = get_clotho_loader(split='development',
                                  input_field_name=input_field_name,
                                  load_into_memory=False,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)

evaluation_data = get_clotho_loader(split='evaluation',
                                    input_field_name=input_field_name,
                                    load_into_memory=False,
                                    batch_size=batch_size,
                                    num_workers=num_workers)

main_logger.info(f'Data loaded.')
main_logger.info(f'Len of training data: {len(training_data)}')
main_logger.info(f'Len of evaluation data: {len(evaluation_data)}')

if config.training.label_smoothing:
    criterion = LabelSmoothingLoss(ntokens, smoothing=0.1)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

epochs = config.training.epochs
ep = 1

if config.mode == 'train':
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
        if epoch >= 15:
            torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    }, str(model_output_dir) + '/model_{}.pt'.format(epoch))
    main_logger.info('Training done.')

if config.mode == 'eval':
    model.load_state_dict(torch.load(config.path.model)['model'])
    eval_greedy(evaluation_data)
    eval_beam(evaluation_data, beam_size=2)
    eval_beam(evaluation_data, beam_size=3)
    eval_beam(evaluation_data, beam_size=4)
    eval_beam(evaluation_data, beam_size=5)
    main_logger.info('Evaluation done')

