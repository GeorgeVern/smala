# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-mono-data.sh $lg
#


set -e

lg=$1  # input language
tknzr=$2  # target tokenizer name

N_THREADS=8

MAX_NLINES=19000000

# data path
MAIN_PATH=$PWD
WIKI_PATH=$PWD/data/mono

# tools paths
TOOLS_PATH=$PWD/tools
UNICODE=$TOOLS_PATH/unicode.py

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl


# Wiki data
WIKI_DUMP_NAME=${lg}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_LINK=https://dumps.wikimedia.org/${lg}wiki/latest/$WIKI_DUMP_NAME

# create Wiki paths
mkdir -p $WIKI_PATH/bz2
mkdir -p $WIKI_PATH/txt

# download Wikipedia dump
echo "Downloading $lg Wikipedia dump from $WIKI_DUMP_LINK ..."
wget -c $WIKI_DUMP_LINK -P $WIKI_PATH/bz2/
echo "Downloaded $WIKI_DUMP_NAME in $WIKI_PATH/bz2/$WIKI_DUMP_NAME"

# extract and tokenize Wiki data
cd $MAIN_PATH
echo "*** Cleaning and tokenizing $lg Wikipedia dump ... ***"
if [ ! -f $WIKI_PATH/txt/$lg/$lg.all.txt ]; then
  mkdir -p $WIKI_PATH/txt/$lg
  python3 $TOOLS_PATH/wikiextractor/wikiextractor/WikiExtractor.py $WIKI_PATH/bz2/$WIKI_DUMP_NAME --processes 8 -q -o - \
  | sed "/^\s*\$/d" \
  | grep -v "^<doc id=" \
  | grep -v "</doc>\$" \
  | perl $REPLACE_UNICODE_PUNCT | perl $NORM_PUNC -l $lg | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -no-escape -threads $N_THREADS -l $lg \
  | python3 $UNICODE \
  > $WIKI_PATH/txt/$lg/$lg.all.txt
fi
echo "*** Tokenized  $lg Wikipedia dump to $WIKI_PATH/txt/$lg/$lg.all.txt ***"

if ! [[ -f $WIKI_PATH/txt/$lg/$lg.train.txt && -f $WIKI_PATH/txt/$lg/$lg.valid.txt ]]; then
  # split into train / valid / test
  echo "*** Split into train / valid ***"
  split_data() {
      get_seeded_random() {
          seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
      };
      NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
      if [[ $NLINES -gt $MAX_NLINES ]]
      then
        NEW_NLINES=$MAX_NLINES
      else
        NEW_NLINES=$NLINES
      fi
      echo "Initial number of lines: $NLINES";
      echo "Final number of lines: $NEW_NLINES";

      NTRAIN=$((NEW_NLINES - 5000));
      NVAL=$((NTRAIN + 5000));
      shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
      shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
  }
  split_data $WIKI_PATH/txt/$lg/$lg.all.txt $WIKI_PATH/txt/$lg/$lg.train.txt $WIKI_PATH/txt/$lg/$lg.valid.txt
fi

if [ $lg != "en" ]; then
  python3 utils/learn_tokenizer.py --tokenizer_name $tknzr --files $WIKI_PATH/txt/$lg/$lg.train.txt  $WIKI_PATH/txt/$lg/$lg.valid.txt
fi
