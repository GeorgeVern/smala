import argparse
from typing import Optional, Tuple
import logging
from tokenizers import BertWordPieceTokenizer
import os

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def save_vocabulary(tokenizer, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    if os.path.isdir(save_directory):
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
    else:
        vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
    vocab_file = save_vocab_dict(vocab_file, tokenizer.get_vocab())
    return vocab_file


def save_vocab_dict(vocab_file, vocab_dict):
    index = 0
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab_dict.items(), key=lambda kv: kv[1]):
            if index != token_index:
                logger.warning(
                    "Saving vocabulary to {}: vocabulary indices are not consecutive."
                    " Please check that the vocabulary is not corrupted!".format(vocab_file)
                )
                index = token_index
            writer.write(token + "\n")
            index += 1
    return (vocab_file,)


def main(args):
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True, lowercase=args.lowercase,
                                       strip_accents=False)
    
    print(args.files)

    tokenizer.train(args.files, vocab_size=30522, min_frequency=2, show_progress=True)

    if os.path.isdir(args.tokenizer_dir):
        raise ValueError("Directory already exists.")
    else:
        os.makedirs(args.tokenizer_dir)

    save_vocabulary(tokenizer, args.tokenizer_dir + 'vocab.txt')
    tokenizer.save(args.tokenizer_dir + 'tokenizer.json', pretty=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate target embeddings from alignments')
    parser.add_argument('--tokenizer_dir',
                        help='where to store the learned tokenizer')
    parser.add_argument('--lowercase', default=True, help='whether to lowercase data')
    parser.add_argument('--files', nargs='+', default=['/data/mono/txt/en/en.train.txt',
                                                       '/data/mono/txt/en/en.valid.txt'],
                        help='where the input files are located')
    args = parser.parse_args()
    main(args)
