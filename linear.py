"""
Linear Evaluation script.
Use config/*.yaml by changing it appropriately
"""

import sys
sys.path.insert(0, '.')
from models import SimCLR, SimSiam
from util.test import test_all_datasets
import numpy as np
from datetime import datetime
import torch
import os
from config.option import Options
from util.utils import summary_writer, logger
from util.utils import log
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(10)
torch.manual_seed(10)


if __name__ == '__main__':
    args = Options().parse()
    log_dir = os.path.dirname(os.path.abspath(args.eval.model_path))
    _, checkpoint = os.path.split(args.eval.model_path)
    writer = summary_writer(args, log_dir, checkpoint + '_Evaluation')
    logger(args, checkpoint + '{}_test.log'.format(args.eval.dataset.name))
    args.start_time = datetime.now()
    log("Starting testing of SSL model at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))

    if args.eval.model == 'simclr':
        model = SimCLR(args, args.eval.dataset.img_size, backbone=args.eval.backbone)
    elif args.eval.model == 'simsiam':
        model = SimSiam(args, args.eval.dataset.img_size, backbone=args.eval.backbone)

    state_dict = torch.load(args.eval.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.cuda()
    test_all_datasets(args, writer, model)
    writer.close()
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)