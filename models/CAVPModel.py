from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .BasicModel import BasicModel
from functools import reduce


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        masks_list = list(att_masks.data.long().sum(1))
    else:
        masks_list = [att_feats.size(1)] * att_feats.size(0)

    packed = pack_padded_sequence(att_feats, masks_list, batch_first=True)
    return pad_packed_sequence(PackedSequence(module(packed[0]), packed[1]), batch_first=True)[0]


class CAVPFrame(BasicModel):
    def __init__(self, opt):
        super(CAVPFrame, self).__init__()
        self.vocab_size = opt.vocab_size
        self.word_embed_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.word_embed_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.ind_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.ind_proj = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, bsz, self.rnn_size).zero_(),
                weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _forward(self, fc_feats, roi_feats, seq, att_masks=None):
        roi_feats, att_masks = self.clip_att(roi_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.data.new(batch_size, seq.size(1) - 1, self.vocab_size + 1).zero_()

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        ind_feats = pack_wrapper(self.ind_embed, roi_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_ind_feats = self.ind_proj(ind_feats)
        context_feats = fc_feats.unsqueeze(1)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.word_embed(it)

            output, state, output_feat = self.core(xt, fc_feats, ind_feats,
                                                 p_ind_feats, context_feats, state, att_masks)
            context_feats = torch.cat([context_feats, output_feat.unsqueeze(1)], dim=1)

            output = F.log_softmax(self.logit(output), dim=1)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, context_feats, tmp_fc_feats, tmp_ind_feats, tmp_p_ind_feats,
                            tmp_att_masks, state):
        # 'it' contains a word index
        xt = self.word_embed(it)

        output, state, output_feat = self.core(xt, tmp_fc_feats, tmp_ind_feats, tmp_p_ind_feats,
                                                 context_feats, state, tmp_att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state, output_feat

    def _sample_beam(self, fc_feats, roi_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        ind_feats = pack_wrapper(self.ind_embed, roi_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_ind_feats = self.ind_proj(ind_feats)
        context_feats = fc_feats.unsqueeze(1)

        assert beam_size <= self.vocab_size + 1, 'otherwise this corner case causes a few headaches down the road.'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seq_logprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_ind_feats = ind_feats[k:k + 1].expand(*((beam_size,) + ind_feats.size()[1:])).contiguous()
            tmp_p_ind_feats = p_ind_feats[k:k + 1].expand(*((beam_size,) + p_ind_feats.size()[1:])).contiguous()
            tmp_context_feats = context_feats[k:k + 1].expand(*((beam_size,) + context_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k + 1].expand(
                *((beam_size,) + att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.word_embed(Variable(it, requires_grad=False))

                output, state, output_feat = self.core(xt, tmp_fc_feats, tmp_ind_feats,
                                                     tmp_p_ind_feats, tmp_context_feats, state, tmp_att_masks)
                tmp_context_feats = torch.cat([tmp_context_feats, output_feat.unsqueeze(1)], dim=1)
                logprobs = F.log_softmax(self.logit(output), dim=1)

            # In beam_search inside, call the get_logprobs_state using input args
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_context_feats, tmp_fc_feats, tmp_ind_feats,
                                                  tmp_p_ind_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seq_logprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seq_logprobs.transpose(0, 1)

    def _sample(self, fc_feats, roi_feats, att_masks=None, opt={}):
        roi_feats, att_masks = self.clip_att(roi_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        ind_feats = pack_wrapper(self.ind_embed, roi_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_ind_feats = self.ind_proj(ind_feats)
        context_feats = fc_feats.unsqueeze(1)

        # seq = []
        # seq_logprobs = []
        seq = fc_feats.data.new(batch_size, self.seq_length).long().zero_()
        seq_logprobs = fc_feats.data.new(batch_size, self.seq_length).zero_()
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sample_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                # gather the logprobs at sampled positions
                sample_logprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                it = it.view(-1).long()  # and flatten indices for downstream processing

            xt = self.word_embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seq_logprobs[:, t - 1] = sample_logprobs.view(-1)

            output, state, output_feat = self.core(xt, fc_feats, ind_feats,
                                                 p_ind_feats, context_feats, state, att_masks)
            context_feats = torch.cat([context_feats, output_feat.unsqueeze(1)], dim=1)

            if decoding_constraint and t > 0:
                tmp = output.data.new(output.size(0), self.vocab_size + 1).zero_()
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = F.log_softmax(self.logit(output) + tmp, dim=1)
            else:
                logprobs = F.log_softmax(self.logit(output), dim=1)

        return seq, seq_logprobs


class CAVPCore(nn.Module):
    def __init__(self, opt):
        super(CAVPCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.scco_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)

        self.single_sp = SingleSP(opt)
        self.context_sp = ContextSP(opt)
        self.comp_sp = CompSP(opt)
        self.output_sp = OutputSP(opt)

    def forward(self, xt, fc_feats, roi_feats, p_roi_feats, context_feats, state, att_masks=None):
        prev_h = state[0][-1]
        env = torch.cat([prev_h, fc_feats, xt], 1)
        h_vis, c_vis = self.scco_lstm(env, (state[0][0], state[1][0]))

        # Compute features
        single_feat = self.single_sp(h_vis, roi_feats, p_roi_feats, att_masks)
        context_feat = self.context_sp(h_vis, context_feats, att_masks)
        comp_feat = self.comp_sp(h_vis, roi_feats, context_feat, att_masks)
        output_feats = self.output_sp(h_vis, single_feat, comp_feat, fc_feats)

        # language LSTM
        lang_lstm_input = torch.cat([output_feats, h_vis], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_vis, h_lang]), torch.stack([c_vis, c_lang]))

        return output, state, output_feats


class ComputeRelation(nn.Module):
    def __init__(self, opt):
        super(ComputeRelation, self).__init__()
        self.rnn_size = opt.rnn_size
        self.proj = nn.Linear(self.rnn_size*2, self.rnn_size)

    def forward(self, roi_feats, context_feat, att_masks=None):
        context_feats = context_feat.unsqueeze(1).expand_as(roi_feats)
        return context_feats - roi_feats


class ContextSP(nn.Module):
    def __init__(self, opt):
        super(ContextSP, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.wh = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wv = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wa = nn.Linear(opt.att_hid_size, 1)

        self.compute_relation = ComputeRelation(opt)

    def forward(self, h, context_feats, att_masks=None):
        feats = context_feats
        feats_ = self.wv(feats)

        dot = self.wh(h).unsqueeze(1).expand_as(feats_) + feats_
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        context_feat = torch.bmm(weight.unsqueeze(1), feats).squeeze(1)

        return context_feat


class SingleSP(nn.Module):
    def __init__(self, opt):
        super(SingleSP, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.wh = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wa = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, roi_feats, p_roi_feats, att_masks=None):
        dot = self.wh(h).unsqueeze(1).expand_as(p_roi_feats) + p_roi_feats
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        if att_masks is not None:
            weight = weight * att_masks
            weight = weight / weight.sum(1, keepdim=True)

        single_feat = torch.bmm(weight.unsqueeze(1), roi_feats).squeeze(1)

        return single_feat


class CompSP(nn.Module):
    def __init__(self, opt):
        super(CompSP, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.wh = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wv = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wa = nn.Linear(opt.att_hid_size, 1)

        self.compute_relation = ComputeRelation(opt)

    def forward(self, h, roi_feats, context_feat, att_masks=None):
        feats = self.compute_relation(roi_feats, context_feat, att_masks)
        feats_ = pack_wrapper(self.wv, feats, att_masks)

        dot = self.wh(h).unsqueeze(1).expand_as(feats_) + feats_
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        if att_masks is not None:
            weight = weight * att_masks
            weight = weight / weight.sum(1, keepdim=True)

        comp_feat = torch.bmm(weight.unsqueeze(1), feats).squeeze(1)

        return comp_feat


class OutputSP(nn.Module):
    def __init__(self, opt):
        super(OutputSP, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.wv = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wh = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wa = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, single_feat, comp_feat, fc_feats):
        feats = torch.stack([single_feat, comp_feat, fc_feats], dim=1)
        feats_ = self.wv(feats)

        dot = self.wh(h).unsqueeze(1).expand_as(feats_) + feats_
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        output_feat = torch.bmm(weight.unsqueeze(1), feats).squeeze(1)

        return output_feat


class CAVPModel(CAVPFrame):
    def __init__(self, opt):
        super(CAVPModel, self).__init__(opt)
        self.num_layers = 2
        self.core = CAVPCore(opt)
