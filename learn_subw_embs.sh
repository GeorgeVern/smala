#!/bin/bash

set -e

lg=$1
tknzr=$2

N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs

TOOLS_PATH=$PWD/tools

# tokenized data path
TGT_WP=$PWD/data/mono/txt/$lg/WP/$lg.train.wp

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText-0.9.2
FASTTEXT=$FASTTEXT_DIR/fasttext

#
# Learn tokenizer from the data and use it to segment data
#
if [[ $lg = "en" ]]; then
  tknzr="bert"
fi
python3 utils/apply_tokenizer.py --tokenizer $tknzr --file "data/mono/txt/$lg/$lg.train.txt"

#
# Train fastText on source and target embeddings seperately
#
if ! [[ -f "$TGT_WP.vec" ]]; then
  echo "Training fastText on $TGT_WP..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 1024 -thread $N_THREADS -ws 5 -neg 10 -input $TGT_WP -output $TGT_WP
fi
echo "Target embeddings in: $TGT_WP.vec"
