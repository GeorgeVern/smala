# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os

import torch
import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertTokenizerFast
import numpy as np

parser = argparse.ArgumentParser('generate target embeddings from alignments')
parser.add_argument('--tgt_vocab',
                    default='../alignments/en-el/new_tgt_vocab.txt',
                    help='target vocabulary file')
parser.add_argument('--src_model',
                    default='bert-base-uncased',
                    help='source pre-trained file')
parser.add_argument('--cache_dir', default='../cache',
                    help='where the pretrained models downloaded from huggingface.com are stored')
parser.add_argument('--prob',
                    default='../alignments/en-el/probs_vector.pth',
                    help='subword translation probability')
parser.add_argument('--tgt_model',
                    default='../emb_layer/el/bert-ours_align_embs',
                    help='save the target model')
params = parser.parse_args()
print(params)


def guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=None):
    emb_dim = src_embs.size(1)
    num_tgt = tgt_tokenizer.vocab_size

    # init with zero
    tgt_embs = src_embs.new_empty(num_tgt, emb_dim)
    tgt_bias = src_bias.new_zeros(num_tgt)
    nn.init.normal_(tgt_embs, mean=0, std=emb_dim ** -0.5)

    # copy over embeddings of special words
    for i in src_tokenizer.all_special_tokens:
        print(i, tgt_tokenizer.vocab[i], src_tokenizer.vocab[i])
        tgt_embs[tgt_tokenizer.vocab[i]] = src_embs[src_tokenizer.vocab[i]]
        tgt_bias[tgt_tokenizer.vocab[i]] = src_bias[src_tokenizer.vocab[i]]

    # initialize randomly
    if prob is None:
        print('| INITIALIZE EMBEDDINGS AND BIAS RANDOMLY')
        return tgt_embs, tgt_bias

    num_src_per_tgt = np.array([len(x) for x in prob.values()]).mean()
    print(f'| # aligned src / tgt: {num_src_per_tgt:.5}')

    for t, ws in prob.items():
        if (t not in tgt_tokenizer.vocab.keys()) or (t in tgt_tokenizer.all_special_tokens): continue

        px, ix = [], []
        for e, p in ws.items():
            # get index of the source word e
            j = src_tokenizer.convert_tokens_to_ids(e)
            ix.append(j)
            px.append(p)
        px = torch.tensor(px).to(src_embs.device)
        # get index of target word t
        ti = tgt_tokenizer.vocab[t]
        tgt_embs[ti] = px @ src_embs[ix]
        tgt_bias[ti] = px.dot(src_bias[ix])

    return tgt_embs, tgt_bias


def init_tgt(params):
    """
    Initialize the parameters of the target model
    """
    prob = None
    if params.prob:
        print(' | load word translation probs!')
        prob = torch.load(params.prob)

    print(f'| load English pre-trained model: {params.src_model}')
    config = AutoConfig.from_pretrained(params.src_model, cache_dir=params.cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(
        params.src_model,
        from_tf=bool(".ckpt" in params.src_model),
        config=config,
        cache_dir=params.cache_dir,
    )

    # note that we do lowercase but not strip accents
    src_tokenizer = AutoTokenizer.from_pretrained(params.src_model, cache_dir=params.cache_dir, use_fast=True)

    # get English word-embeddings and bias
    src_embs = model.base_model.embeddings.word_embeddings.weight.detach().clone()
    src_bias = model.cls.predictions.bias.detach().clone()

    # initialize target tokenizer, we always use BertWordPieceTokenizer for the target language
    tgt_tokenizer = BertTokenizerFast(vocab_file=params.tgt_vocab, do_lower_case=True, strip_accents=False)

    tgt_embs, tgt_bias = guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=prob)

    # checksum for debugging purpose
    print(' checksum src | embeddings {:.5f} - bias {:.5f}'.format(
        src_embs.norm().item(), src_bias.norm().item()))
    model.base_model.embeddings.word_embeddings.weight.data = tgt_embs
    model.cls.predictions.bias.data = tgt_bias
    model.tie_weights()
    print(' checksum tgt | embeddings {:.5f} - bias {:.5f}'.format(
        model.base_model.embeddings.word_embeddings.weight.norm().item(),
        model.cls.predictions.bias.norm().item()))

    # save the model
    os.makedirs(params.tgt_model, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(params.tgt_model)


if __name__ == '__main__':
    init_tgt(params)
