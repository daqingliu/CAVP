from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
import logging
import numpy as np

import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../'))

import opts
import models
import eval_utils
from dataloader import *
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward


def initialize_logger(output_file):
    formatter = logging.Formatter("<%(asctime)s.%(msecs)03d> %(message)s",
                                  "%m-%d %H:%M:%S")
    handler = logging.FileHandler(output_file, mode='a')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


def train(opt):
    logger = initialize_logger(os.path.join(opt.checkpoint_path, 'train.log'))
    print = logger.info

    if opt.use_box:
        opt.att_feat_size = opt.att_feat_size + 5
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Print out the option variables
    print("*" * 20)
    for k, v in opt.__dict__.items():
        print("%r: %r" % (k, v))
    print("*" * 20)

    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos.json'), 'r') as f:
            infos = json.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None

    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    start_time = time.time()
    while True:
        if update_lr_flag:
            # Assign the learning rate
            if 0 <= opt.learning_rate_decay_start < epoch:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if 0 <= opt.scheduled_sampling_start < epoch:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer()
            else:
                sc_flag = False

            update_lr_flag = False

        # Load data from train split (0)
        batch_data = loader.get_batch('train')
        torch.cuda.synchronize()

        tmp = [batch_data['fc_feats'], batch_data['att_feats'], batch_data['labels'],
               batch_data['masks'], batch_data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        optimizer.zero_grad()
        if not sc_flag:
            outputs = dp_model(fc_feats, att_feats, labels, att_masks)
            loss = crit(outputs, labels[:, 1:], masks[:, 1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max': 0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, batch_data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data
        torch.cuda.synchronize()

        # Update the iteration and epoch
        iteration += 1
        if batch_data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Print train loss or avg reward
        if iteration % opt.losses_print_every == 0:
            if not sc_flag:
                print("iter {} (epoch {}), loss = {:.3f}, time = {:.3f}"
                      .format(iteration, epoch, loss.item(), time.time() - start_time))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time = {:.3f}"
                      .format(iteration, epoch, np.mean(reward[:, 0]), time.time() - start_time))
            start_time = time.time()

        # make evaluation on validation set, and save model
        if (opt.save_checkpoint_every > 0 and iteration % opt.save_checkpoint_every == 0)\
                or (opt.save_checkpoint_every <= 0 and update_lr_flag):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.simple_eval_split(dp_model, loader, eval_kwargs)

            # Save model if is improving on validation result
            if not os.path.exists(opt.checkpoint_path):
                os.makedirs(opt.checkpoint_path)

            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscellaneous information
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = vars(opt)
            infos['vocab'] = loader.get_vocab()

            with open(os.path.join(opt.checkpoint_path, 'infos.json'), 'w') as f:
                json.dump(infos, f, sort_keys=True, indent=4)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos-best.json'), 'w') as f:
                    json.dump(infos, f, sort_keys=True, indent=4)

            # Stop if reaching max epochs
            if opt.max_epochs != -1 and epoch >= opt.max_epochs:
                break


if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)
