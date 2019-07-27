from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse

import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../'))

import models
from dataloader import *
import eval_utils
import misc.utils as utils

import time


def parse_eval_opt():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Input paths
    parser.add_argument('--log_path', type=str, default='log/log_cavp_01')
    parser.add_argument('--load_best', type=int, default=1)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--sample_max', type=int, default=1)
    parser.add_argument('--max_ppl', type=int, default=0)
    parser.add_argument('--decoding_constraint', type=int, default=1)

    # Basic options
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--language_eval', type=int, default=1)
    parser.add_argument('--dump_json', type=int, default=1)

    # Sampling options
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--diversity_lambda', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=1.0)

    # For evaluation on MSCOCO images from some split:
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_box_36')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label_with_gt.h5')
    parser.add_argument('--input_json', type=str, default='data/coco_with_gt.json')

    # misc
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--verbose_beam', type=int, default=0)
    parser.add_argument('--verbose_loss', type=int, default=0)

    args = parser.parse_args()
    return args


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def split_eval():
    opt = parse_eval_opt()
    # Load infos
    infos_file = 'infos-best.json' if opt.load_best else 'info.json'
    infos_path = os.path.join(opt.log_path, infos_file)
    with open(infos_path, 'r') as f:
        infos = json.load(f)
    infos['opt'] = Bunch(infos['opt'])

    # override and collect parameters
    if len(opt.input_att_dir) == 0:
        opt.input_att_dir = infos['opt'].input_att_dir
        opt.input_label_h5 = infos['opt'].input_label_h5
    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = infos['opt'].batch_size
    if len(opt.id) == 0:
        opt.id = infos['opt'].id

    ignore = ["id", "batch_size", "start_from", "language_eval",
              "input_att_dir", "input_label_h5", "input_json"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    # Print out the option variables
    print("*" * 20)
    for k, v in opt.__dict__.items():
        print("%r: %r" % (k, v))
    print("*" * 20)

    # Setup the model
    model = models.setup(opt)
    model_file = 'model-best.pth' if opt.load_best else 'model.pth'
    model_path = os.path.join(opt.log_path, model_file)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    crit = utils.LanguageModelCriterion()

    # Create the Data Loader instance
    loader = DataLoader(opt)
    loader.ix_to_word = infos['vocab']

    # Set sample options
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))
    print(lang_stats)

    # dump result json
    if opt.dump_json:
        json.dump(split_predictions, open(os.path.join(opt.log_path, opt.split+'_res.json'), 'w'))


if __name__ == '__main__':
    split_eval()
