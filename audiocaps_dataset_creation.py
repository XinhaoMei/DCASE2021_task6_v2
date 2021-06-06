#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


from tools.audiocaps_dataset import create_dataset
from tools.utils import setup_seed


if __name__ == '__main__':
    setup_seed(20)
    create_dataset()
