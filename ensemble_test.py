#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import platform
import sys
import time
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from loguru import logger
from tools.ensemble import EnsembleModel
from tools.file_io import load_pickle_file
from models.TransModel import TransformerModel
from eval_metrics import evaluate_metrics
from tools.beam import beam_decode
from tools.config_loader import get_config
from tools.utils import greedy_decode, decode_output
from data_handling.DataLoader import get_dataloader


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

        if beam_size == 3:
            for metric, values in metrics.items():
                val_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')

        return metrics


def ensemble_validate(data_loader, model, beam_size, sos_ind, eos_ind, vocabulary, log_dir, epoch, device, is_keyword, keywords_list):

    val_logger = logger.bind(indent=1)
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
                output = model.greedy_decode(src, keyword=kw)
            else:
                output = model.beam_decode(src, keyword=kw, beam_width=beam_size)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):  # batch_size
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

        if beam_size == 3:
            for metric, values in metrics.items():
                val_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')

        return metrics


if __name__ == '__main__':

    config = get_config('settings')

    log_output_dir = Path('outputs', 'ensemble', 'logging')
    log_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # wav settings
    if config.dataset == 'AudioCaps':
        config.wav.sr = 32000
        config.wav.window_size = 1024
        config.wav.hop_length = 320
    elif config.dataset == 'Clotho':
        config.wav.sr = 44100
        config.wav.window_size = 1024
        config.wav.hop_length = 512

    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())

    test_loader = get_dataloader('test', config, return_dict=True)

    vocabulary = load_pickle_file(config.path.vocabulary.format(config.dataset))
    ntokens = len(vocabulary)
    sos_ind = vocabulary.index('<sos>')
    eos_ind = vocabulary.index('<eos>')

    num_keywords = config.num_keywords
    test_keywords_dict = load_pickle_file('data/Clotho/pickles/645/test_keywords_dict_pred_{}.p'.format(num_keywords))
    test_size = len(test_loader.dataset)
    test_keywords = np.zeros((test_size, num_keywords))
    for i in range(test_size):
        file_name = test_loader.dataset[i][-1]
        keywords = test_keywords_dict[file_name]
        keywords_index = [vocabulary.index(word) for word in keywords]
        while len(keywords_index) < num_keywords:
            keywords_index.append(keywords_index[-1])
        test_keywords[i] = keywords_index

    model_list = []
    check_points = ['outputs/DCASE/645/pooling/factor3/after/off_ls_5_AdamW_no_keywords_finetune_data_Clotho_pooling_type_avg_seed_40/model/model_ep15.pth',
                    'outputs/DCASE/645/pooling/factor3/after/off_ls_5_AdamW_no_keywords_finetune_data_Clotho_pooling_type_avg_seed_40/model/model_ep18.pth',
                    'outputs/DCASE/645/pooling/factor3/after/off_ls_5_AdamW_no_keywords_finetune_data_Clotho_pooling_type_avg_seed_40/model/model_ep21.pth',
                    'outputs/DCASE/645/pooling/factor3/after/off_ls_5_AdamW_no_keywords_finetune_data_Clotho_pooling_type_avg_seed_40/model/model_ep24.pth',
                    'outputs/DCASE/645/pooling/factor3/after/off_ls_5_AdamW_no_keywords_finetune_data_Clotho_pooling_type_avg_seed_40/model/model_ep27.pth',
                    'outputs/DCASE/645/pooling/factor3/after/off_ls_5_AdamW_no_keywords_finetune_data_Clotho_pooling_type_avg_seed_40/model/model_ep30.pth'
                    ]

    main_logger.info(f'Total {len(check_points)} checkpoints.')

    for i, cp in enumerate(check_points):

        cp_config = torch.load(cp)['config']
        model = TransformerModel(cp_config)
        model = model.to(device)
        model.eval()

        main_logger.info(f'Loading checkpoint for model {i + 1}')

        model.load_state_dict(torch.load(cp)['model'])
        model_list.append(model)

    main_logger.info('Evaluate single model performance')
    for i, model in enumerate(model_list):
        main_logger.info(f'Evaluating for model {i + 1}')
        for beam_size in range(1, 4):
            validate(test_loader, model, beam_size=beam_size, sos_ind=sos_ind,
                     eos_ind=eos_ind, vocabulary=vocabulary, log_dir=log_output_dir,
                     epoch=i, device=device, is_keyword=model.is_keywords, keywords_list=test_keywords)

    main_logger.info('Start ensemble.')
    ensemble_model = EnsembleModel(model_list, sos_ind=sos_ind, eos_ind=eos_ind)
    for beam_size in range(1, 4):
        ensemble_validate(test_loader, ensemble_model, beam_size=beam_size, sos_ind=sos_ind,
                          eos_ind=eos_ind, vocabulary=vocabulary, log_dir=log_output_dir,
                          epoch=0, device=device, is_keyword=True, keywords_list=test_keywords)
    main_logger.info('Ensemble done.')
