#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import sys
import torch
import torch.nn as nn
import numpy as np
import time
from loguru import logger
import argparse
from tqdm import tqdm
from pathlib import Path
from data_handling.clotho_dataset import get_clotho_loader
from tools.config_loader import get_config
from tools.utils import setup_seed, align_word_embedding, \
set_tgt_padding_mask, rotation_logger, \
decode_output, beam_search, greedy_decode
from tools.file_io import load_picke_file
from models.TransModel import TransformerModel
from pprint import PrettyPrinter
from tools.rl_utils import scst_sample, get_self_critical_reward
from eval_metrics import evaluate_metrics
from warmup_scheduler import GradualWarmupScheduler


def train():
    start_time = time.time()
    batch_losses = torch.zeros(len(training_data))
    max_len = 30

    for batch_idx, train_batch in tqdm(enumerate(training_data), total=len(training_data)):

        src, tgt, f_names, tgt_len, captions = train_batch
        src = src.to(device)
        tgt = tgt.to(device)

        model.eval()
        with torch.no_grad():
            if config.rl.mode == 'beam':
                output = beam_search(model, src, beam_size=2)
                out_tensor = torch.zeros(len(output), max_len).fill_(eos_ind)

                for i in range(len(output)):
                    tmp = torch.zeros(max_len - output[i][1:].size()[0]).fill_(eos_ind).long().to(device)
                    tmp = torch.cat((output[i][1:], tmp), dim=0)
                    out_tensor[i] = tmp
            elif config.rl.mode == 'greedy':
                output = greedy_decode(model, src).int()
                out_tensor = torch.zeros(output.shape).fill_(eos_ind).to(device)
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        out_tensor[i, j] = output[i, j]
                        if output[i, j] == eos_ind:
                            break

        model.train()
        sample, sampled_logprobs = scst_sample(model, src, tgt, sos_ind=sos_ind, eos_ind=eos_ind)

        optimizer.zero_grad()
        reward_score, sample_score = get_self_critical_reward(out_tensor, sample, tgt, words_list, sos_ind, eos_ind)
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
        cider = greedy_metrics['Cider']['score']
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
        cider = beam_metrics['cider']['score']
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')
        spiders.append(spider)
        if beam_size == 3 and (epoch % 5) == 0:
            for metric, values in beam_metrics.items():
                main_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')
        spiders.append(spider)
        if spider >= max(spiders):
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": beam_size,
                        "epoch": epoch},
                        str(model_output_dir) + '/best_model.pt'.format(epoch))


parser = argparse.ArgumentParser(description='Finetune using Reinforcement learning')

parser.add_argument('-n', '--exp_name', type=str, default='exp1', help='name of the experiment')

args = parser.parse_args()

config = get_config()

setup_seed(config.training.seed)

exp_name = args.exp_name

model_output_dir = Path('rl_outputs', exp_name, 'model')
log_output_dir = Path('rl_outputs', exp_name, 'logging')

model_output_dir.mkdir(parents=True, exist_ok=True)
log_output_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}',
           level='INFO', filter=lambda record: record['extra']['indent'] == 1)

logger.add(log_output_dir.joinpath('output.txt'),
           format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 1)

logger.add(str(log_output_dir) + '/captions.txt',
           format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 2,
           rotation=rotation_logger)

logger.add(str(log_output_dir) + '/beam_captions.txt',
           format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 3,
           rotation=rotation_logger)

main_logger = logger.bind(indent=1)

printer = PrettyPrinter()

device, device_name = (torch.device('cuda'), torch.cuda.get_device_name(torch.cuda.current_device()))

main_logger.info('Finetune using Reinforcement Learning.')
main_logger.info(f'Process on {device_name}')

words_list_path = config.path.clotho.words_list
# words_list_path = 'data/pickles/new_words_list.p'
words_list = load_picke_file(words_list_path)
ntokens = len(words_list)
sos_ind = words_list.index('<sos>')
eos_ind = words_list.index('<eos>')

pretrained_model = torch.load(config.rl.model)['model']

model = TransformerModel(config, words_list)
model.to(device)

model.load_state_dict(pretrained_model)

main_logger.info('Training settings:\n'
                 f'{printer.pformat(config)}')

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

# validation_data = get_clotho_loader(split='validation',
#                                     input_field_name=input_field_name,
#                                     load_into_memory=False,
#                                     batch_size=batch_size,
#                                     num_workers=num_workers)

evaluation_data = get_clotho_loader(split='evaluation',
                                    input_field_name=input_field_name,
                                    load_into_memory=False,
                                    batch_size=batch_size,
                                    num_workers=num_workers)

main_logger.info(f'Data loaded.')
main_logger.info(f'Len of training data: {len(training_data)}')
main_logger.info(f'Len of evaluation data: {len(evaluation_data)}')

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.rl.lr, weight_decay=1e-6)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

# optimizer.zero_grad()
# optimizer.step()

epochs = config.rl.epochs
ep = 1
spiders = []

epoch = 0
main_logger.info('Evaluating baseline performance...')
eval_greedy(evaluation_data)
eval_beam(evaluation_data, beam_size=2)
eval_beam(evaluation_data, beam_size=3)
eval_beam(evaluation_data, beam_size=4)
eval_beam(evaluation_data, beam_size=5)

main_logger.info(f'Optimize {config.rl.mode} search.')

for epoch in range(ep, epochs + 1):

    # scheduler_warmup.step(epoch)

    main_logger.info(f'Finetune using RL epoch {epoch}...')
    train()
    main_logger.info('Evaluating...')
    eval_greedy(evaluation_data)
    eval_beam(evaluation_data, beam_size=2)
    eval_beam(evaluation_data, beam_size=3)
    eval_beam(evaluation_data, beam_size=4)
    eval_beam(evaluation_data, beam_size=5)
main_logger.info('Finetune done.')
