"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

from random import randint, shuffle
from random import random as rand
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter

import tokenization
import models
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair, _sample_mask
from albertgen import get_ALBERT_datagen
from collections import namedtuple


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 2)

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        embed_weight2 = self.transformer.embed.tok_embed2.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()

        ## project embedding layer to vocabulary layer
        embed_weight1 = self.transformer.embed.tok_embed1.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1

        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder2(self.decoder1(h_masked)) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf


FullCFG = namedtuple("FullCFG", ["args", "model", "pre"])


def main(args):
    model_cfg = models.Config.from_json(args.model_cfg)
    pre_cfg = train.Config.from_json(args.train_cfg)
    cfg = FullCFG(args, model_cfg, pre_cfg) 

    #set_seeds(cfg.pre.seed)

    data_iter = get_ALBERT_datagen(cfg)
    
    #for s in data_iter:
        #print(s)
        #break
    
    #return

    model = BertModel4Pretrain(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim4GPU(pre_cfg, model)
    trainer = train.Trainer(pre_cfg, model, data_iter, optimizer, args.save_dir, get_device())

    cur_date = datetime.now().strftime("%Y.%m.%d_%H:%M")
    log_dir = f"{args.log_dir}/{cur_date}"
    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_sop = criterion2(logits_clsf, is_next) # for sentence classification
        batch_acc_lm = (torch.argmax(logits_lm, 2) == masked_ids).float().mean()
        batch_acc_sop = (torch.argmax(logits_clsf, 1) == is_next).float().mean()
        #print(batch_acc_sop.item())
        
        if global_step % 10 == 0:
            writer.add_scalars(
                'data/losses', {
                    'loss_lm': loss_lm.item(),
                    'loss_sop': loss_sop.item(),
                    'loss_total': (loss_lm + loss_sop).item(),
                }, global_step)
                            
            writer.add_scalars(
                'data/learning_rate', {
                    'lr': optimizer.get_lr()[0]
                }, global_step)
            
            writer.add_scalars(
                'data/batch_accuracy', {
                    'acc_lm': batch_acc_lm.item(),
                    'acc_sop': batch_acc_sop.item()
                }, global_step)
        
        return loss_lm + loss_sop

    trainer.train(get_loss, model_file=None, data_parallel=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ALBERT Language Model')
    parser.add_argument('--data_file', type=str, default='./data/wiki.train.tokens')
    parser.add_argument('--vocab', type=str, default='./data/vocab.txt')
    parser.add_argument('--train_cfg', type=str, default='./config/pretrain.json')
    parser.add_argument('--model_cfg', type=str, default='./config/albert_unittest.json')

    # official google-reacher/bert is use 20, but 20/512(=seq_len)*100 make only 3% Mask
    # So, using 76(=0.15*512) as `max_pred`
    parser.add_argument('--max_pred', type=int, default=76, help='max tokens of prediction')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='masking probability')

    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    parser.add_argument('--mask_alpha', type=int,
                        default=4, help="How many tokens to form a group.")
    parser.add_argument('--mask_beta', type=int,
                        default=1, help="How many tokens to mask within each group.")
    parser.add_argument('--max_gram', type=int,
                        default=3, help="number of max n-gram to masking")

    parser.add_argument('--tok_workers', type=int,
                        default=1, help="number of preprocessing workers")

    parser.add_argument('--save_dir', type=str, default='./saved')
    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()
    main(args=args)
