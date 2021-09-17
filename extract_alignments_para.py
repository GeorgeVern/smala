import argparse

import os
import torch
from transformers import AutoTokenizer, BertTokenizerFast


def main(args):
    src_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache', use_fast=True,
                                                  do_lower_case=True)
    tgt_tokenizer = BertTokenizerFast(vocab_file=os.path.join("tknzr", args.tgt_tokenizer), do_lower_case=True, strip_accents=False)
    eng_words = list(src_tokenizer.vocab.keys())
    for_words = list(tgt_tokenizer.vocab.keys())

    prob = torch.load(args.similarity_matrix)

    alignment_scores = {}
    alignment_dict = {}
    for key, value in prob.items():
        candidate_alignment = max(prob[key], key=prob[key].get)
        if (key in eng_words) and (key not in alignment_dict) and (candidate_alignment in for_words):
            if max(prob[candidate_alignment], key=prob[candidate_alignment].get) == key:
                alignment_dict[key] = candidate_alignment
                alignment_scores[key, candidate_alignment] = (prob[key][candidate_alignment] +
                                                              prob[candidate_alignment][key]) / 2
    print("Number of aligned subwords: {}".format(len(alignment_dict)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('extract subword alignments')
    parser.add_argument('--tgt_tokenizer', help='tokenizer name')
    parser.add_argument('--similarity_matrix', default='alignments/en-el_fastalign/prob_vector',
                        help='source (english) embeddings')
    args = parser.parse_args()
    main(args)
