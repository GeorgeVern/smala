import argparse
import os
from transformers import BertTokenizerFast, AutoTokenizer


def main(args):
    if args.tokenizer == "bert":
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', cache_dir='cache', use_fast=True,
            do_lower_case=True)
    else:
        tokenizer = BertTokenizerFast(
            vocab_file=args.tokenizer, do_lower_case=True,
            strip_accents=False)

    print("Using tokenizer: {}".format(args.tokenizer))

    with open(args.file) as f:
        text = f.readlines()

    # create an additional directory ../lang/WP/
    filename = args.file.split("/")[-1]
    output_dir = "/".join(args.file.split("/")[:-1])+"/WP"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, ".".join(filename.split(".")[:-1] + ["wp"]))

    with open(output_file, "w+") as output:
        # with open('{}/sampled/all.{}.tok.wp'.format(data_path, language), "w+") as output:
        for line in text:
            output.write('{}\n'.format(" ".join(tokenizer.tokenize(line))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate target embeddings from alignments')
    parser.add_argument('--tokenizer',
                        help='where the pretrained tokenizer is stored')
    parser.add_argument('--file', default='/data/mono/wiki/txt/en/en.train.txt',
                        help='where the file-to-be-tokenized is located')
    args = parser.parse_args()
    main(args)
