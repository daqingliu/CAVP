from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable

import os
import sys
import json
import numpy as np
from json import encoder

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../'))
sys.path.append(os.path.join(this_dir, '../coco-caption'))

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import misc.utils as utils


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] == 0:
            break
        out += str(arr[i]) + ' '
    return out.strip()


def get_score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def simple_eval_split(model, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')

    loader.reset_iterator(split)
    model.eval()

    all_gts = []
    all_res = []
    n = 0

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp

        with torch.no_grad():
            greedy_res, _ = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')

        greedy_res = greedy_res.data.cpu().numpy()
        for i in range(greedy_res.shape[0]):
            all_res.append([array_to_str(greedy_res[i])])

        for i in range(len(data['gts'])):
            all_gts.append([array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))])

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)

        if verbose:
            print('evaluating validation performance... %d/%d' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if 0 <= num_images <= n:
            break

    # After get result, compute scores
    model.train()

    res__ = {i: all_res[i] for i in range(len(all_res))}
    gts = {i: all_gts[i] for i in range(len(all_gts))}

    final_scores = get_score(gts, res__)

    # print out scores
    print('Bleu_1:\t', final_scores['Bleu_1'])
    print('Bleu_2:\t', final_scores['Bleu_2'])
    print('Bleu_3:\t', final_scores['Bleu_3'])
    print('Bleu_4:\t', final_scores['Bleu_4'])
    print('METEOR:\t', final_scores['METEOR'])
    print('ROUGE_L:', final_scores['ROUGE_L'])
    print('CIDEr:\t', final_scores['CIDEr'])

    return 0, 0, final_scores


def language_eval(dataset, preds, model_id, split):
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    ann_file = 'coco-caption/annotations/captions_val2014.json'

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('log/eval_results'):
        os.mkdir('log/eval_results')
    cache_path = os.path.join('log/eval_results/', model_id + '_' + split + '.json')

    coco = COCO(ann_file)
    valid = coco.getImgIds()
    json.dump(preds, open(cache_path, 'w'))  # serialize to temporary json file

    coco_res = coco.loadRes(cache_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    print(len(set(coco_res.getImgIds()) & set(coco.getImgIds())))
    coco_eval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in coco_eval.eval.items():
        out[metric] = score

    img_to_eval = coco_eval.imgToEval
    # for p in preds:
    #     image_id, caption = p['image_id'], p['caption']
    #     img_to_eval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'img_to_eval': img_to_eval}, outfile)

    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join(
                    [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            if verbose:
                print('image %s: ' % (data['infos'][k]['id']), sent.encode('utf8', 'replace'))
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation performance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if 0 <= num_images <= n:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
