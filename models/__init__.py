from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch


def setup(opt):
    if opt.caption_model == 'cavp':
        from .CAVPModel import CAVPModel
        model = CAVPModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(
            opt.start_from, "infos.json")), "infos.json file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model
