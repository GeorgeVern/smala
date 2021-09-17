#!/usr/bin/env python

lg1=$1
lg2=$2

LG1_WP_TOK=$3
LG2_WP_TOK=$4

OUTPUT_DIR=$4

mkdir -p $OUTPUT_DIR

if [ ! -f "$OUTPUT_DIR/text.$lg1-$lg2" ]; then
	:|paste -d ' ||| ' $SRC_WP_TOK - - - - $TGT_WP_TOK > $OUTPUT_DIR/text.$lg1-$lg2
fi 
echo "Bitext saved in $OUTPUT_DIR/text.$lg1-$lg2"

if [ ! -f "$OUTPUT_DIR/cleaned.$lg1-$lg2" ]; then
	python3 utils/clean_bitext.py --bitxt $OUTPUT_DIR/text.$lg1-$lg2 --save $OUTPUT_DIR/cleaned.$lg1-$lg2
fi 
echo "Cleaned bitext saved in $OUTPUT_DIR/cleaned.$lg1-$lg2"

if [ ! -f "$OUTPUT_DIR/align.$lg1-$lg2" ]; then
	tools/fast_align/build/fast_align -i $OUTPUT_DIR/cleaned.$lg1-$lg2 -d -o -v -I 10 > $OUTPUT_DIR/forward.$lg1-$lg2
	tools/fast_align/build/fast_align -i $OUTPUT_DIR/cleaned.$lg1-$lg2 -d -o -v -r -I 10 > $OUTPUT_DIR/reverse.$lg1-$lg2
	tools/fast_align/build/atools -i $OUTPUT_DIR/forward.$lg1-$lg2 -j $OUTPUT_DIR/reverse.$lg1-$lg2 -c grow-diag-final-and > $OUTPUT_DIR/align.$lg1-$lg2
fi
echo "Alignments computed using fast-align in $OUTPUT_DIR/align.$lg1-$lg2"
