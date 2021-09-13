import argparse
from transformers import BertTokenizerFast, AutoTokenizer


def main(args):
    if args.tokenizer == "bert":
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', cache_dir='', use_fast=True,
            do_lower_case=True)
    else:
        tokenizer = BertTokenizerFast(
            vocab_file=args.tokenizer, do_lower_case=True,
            strip_accents=False)

    print("Using tokenizer :{}".format(args.tokenizer))

    with open(args.file) as f:
        text = f.readlines()

    output_file = ".".join(args.file.split(".")[:-1] + ["wp"])

    with open(output_file, "w+") as output:
        for line in text:
            output.write('{}\n'.format(" ".join(tokenizer.tokenize(line))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate target embeddings from alignments')
    parser.add_argument('--tokenizer',
                        help='where the pretrained tokenizer is stored')
    parser.add_argument('--file', nargs='+', default='/data/mono/wiki/txt/en/en.train.txt',
                        help='where the input files are located')
    args = parser.parse_args()
    main(args)
