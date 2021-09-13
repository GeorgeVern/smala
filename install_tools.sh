# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

# data path
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools

# tools
MOSES_DIR=$TOOLS_PATH/mosesdecoder
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# tools path
mkdir -p $TOOLS_PATH

#
# Download and install tools
#

cd $TOOLS_PATH

# Download Moses
if [ ! -d "$MOSES_DIR" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
  # use this to avoid permission denied when executing a mosesdecoder script
  chmod -R 757 $MOSES_DIR
fi

# Download WikiExtractor
if [ ! -d $TOOLS_PATH/wikiextractor ]; then
    echo "Cloning WikiExtractor from GitHub repository..."
    git clone https://github.com/attardi/wikiextractor.git
fi

# Download FastText 
if [ ! -d $TOOLS_PATH/fastText ]; then
    #git clone https://github.com/facebookresearch/fastText.git
    #cd fastText
    #mkdir build && cd build && cmake ..
    #make && make install
    wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
    unzip v0.9.2.zip && rm v0.9.2.zip
    cd fastText-0.9.2
    make
    cd ..
fi

# Download FastAlign 
if [ ! -d $TOOLS_PATH/fast_align ]; then
    git clone https://github.com/clab/fast_align.git
    cd fast_align
    mkdir build
    cd build
    cmake ..
    make
fi
