import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.auto.tokenization_auto import BertTokenizerFast
import numpy as np

from learn_tokenizer import save_vocab_dict


# With this script we create new vocabs/tokenizers that correctly index the new embedding layers for each language:
# L1_embs = [non_shared_l1_subwords, shared_subwords] & L2_embs = [non_shared_l2_subwords, shared_subwords]

def main(args):
    src_tokenizer = AutoTokenizer.from_pretrained(args.src_model, cache_dir=args.cache_dir, use_fast=True)
    
    tgt_vocab = os.path.joint("tknzr", args.tgt_tokenizer, "vocab.txt")
    tgt_tokenizer = BertTokenizerFast(vocab_file=tgt_vocab, do_lower_case=True, strip_accents=False)

    new_tgt_vocab = {}
    src_shared_idx = []
    if args.model_type == "ours":
        with open(args.alignment_dir + 'alignment_dict.json', 'r') as fp:
            alignment_dict = json.load(fp)
        # read the alignment dict and save the indexes of the aligned subword in the source embedding layer
        # create the new target vocab where the aligned target subwords are in the same indexes as in source
        for src_subw, tgt_subw in tqdm(alignment_dict.items()):
            if (src_subw != "</s>") and (src_subw not in src_tokenizer.all_special_tokens) and (
                    tgt_subw not in tgt_tokenizer.all_special_tokens):
                src_subw_index = src_tokenizer.vocab[src_subw]
                new_tgt_vocab[tgt_subw] = src_subw_index
                src_shared_idx.append(src_subw_index)
    elif args.model_type == "joint":
        for subw, index in src_tokenizer.vocab.items():
            if (subw in tgt_tokenizer.vocab) and (subw not in src_tokenizer.all_special_tokens) and (subw != "</s>"):
                src_shared_idx.append(index)
                new_tgt_vocab[subw] = index
    else:
        raise ValueError("No model type provided")
    # find the indexes that are not occupied by shared subwords
    non_shared_mask = np.ones(tgt_tokenizer.vocab_size, dtype=bool)
    non_shared_mask[src_shared_idx] = False
    non_shared_tgt_idx = np.arange(tgt_tokenizer.vocab_size)[non_shared_mask]
    print(len(new_tgt_vocab) + len(non_shared_tgt_idx), len(tgt_tokenizer.get_vocab()))
    # create new vocabularies/tokenizers for the languages so that they agree with the new embeddings
    idx = 0
    # the non-aligned subwords can be put in any order in any of the non-occupied indexes
    for subw in sorted(tgt_tokenizer.get_vocab(), key=tgt_tokenizer.get_vocab().get):
        if subw not in new_tgt_vocab:
            new_tgt_vocab[subw] = non_shared_tgt_idx[idx]
            idx += 1
    if len(src_shared_idx) + idx != tgt_tokenizer.vocab_size:
        raise ValueError("There is a mismatch between the original and the new dictionary")

    os.makedirs(args.alignment_dir, exist_ok=True)
    with open(args.alignment_dir + "src_shared_idx.json", 'w') as f:
        json.dump(src_shared_idx, f, indent=2)
    save_vocab_dict(args.alignment_dir + 'new_tgt_vocab.txt', new_tgt_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('create new vocabulary from the alignment')
    parser.add_argument('--tgt_tokenizer',
                        default='el-tokenizer',
                        help='target vocabulary file')
    parser.add_argument('--src_model', default='bert-base-uncased', help='source pre-trained file')
    parser.add_argument('--cache_dir', default='../cache',
                        help='where the pretrained models downloaded from huggingface.com are stored')
    parser.add_argument('--model_type', default='ours', help='The model type.')
    parser.add_argument('--alignment_dir',
                        default='../alignments/en-el/',
                        help='The path from which to read alignment dict and store the new vocabs')
    args = parser.parse_args()
    print(args)

    main(args)
