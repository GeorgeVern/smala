# *SMALA* - *S*ubword *M*apping and *A*nchoring across *La*nguages
This repository contains source code for our EMNLP 2021 Findings paper: Subword Mapping and Anchoring Across Languages.

## Overview
In our paper we propose a novel method to construct bilingual subword vocabularies. We identify _false positives_ (identical subwords with different meanings across languages) and _false negatives_ (different subwords with similar meanings) as limitation of jointly constructed subword vocabularies. SMALA extracts subword alignments using an unsupervised state-of-the-art mapping technique and uses them to create cross-lingual anchors based on subword similarities.

## Method
We first learn **subwords** separately for each language and then train the corresponding embeddings. We then apply a **mapping** method to obtain similarity scores between the embeddings, which we use to extract **alignments** between subwords of the two languages. We finally tie the parameters of the aligned subwords to create **anchors** during training. 

<!-- SMALA  outperforms current methods for joint construction of multilingual subword vocabulariesin cases where there is no cross-lingual signal, such as zero-shot transfer to an unseen language (XNLI) only by sharing subword embeddings. When cross-lingual supervision is available, SMALA is a viable alternative to create shared bilingual vocabularies. -->

## Prerequisites
### Dependencies
* Python 3.7.9
* [Pytorch](https://pytorch.org/) (tested on 1.6.0)
* [FastText](https://github.com/facebookresearch/fastText)
* [FastAlign](https://github.com/clab/fast_align)
* [VecMap](https://github.com/artetxem/vecmap)
* [Transformers](https://huggingface.co/transformers/) (tested on 4.1.0)
* [Tokenizers](https://github.com/huggingface/tokenizers) (tested on 0.9.4)


### Install Requirements
*Create Environment (Optional):* Ideally, you should create an environment for the project.

    conda create -n smala_env python=3.7.9
    conda activate smala_env
Install PyTorch `1.6.0`:

    conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
    
Clone the project:

```
git clone https://github.com/GeorgeVern/smala.git
cd smala
```

Then install the rest of the requirements:

    pip install -r requirements.txt

### Install tools
Install tools (*) necessary for data extraction, preprocessing and alignment:
    
    bash install tools.sh

(*) You will have to change line 66 from the wikiextractor/WikiExtractor.py script: `from .extract` -> `from extract` otherwise you will get a relative import error.

## SMALA
### Download data
Download and preprocess wikipedia data for **English** (en) and another language, e.g. **Greek** (el):
    
    bash get-mono-data.sh en
    bash get-mono-data.sh el
    
### 1) Subword Mapping
Learn language-speific tokenizer and get subword embeddings for each language:
    
    bash learn_subw_embs.sh en
    bash learn_subw_embs.sh el

Map the monolingual subword embedding into a common space using the **unsupervised** version (currently default) of  VecMap. Clone its github repo ([VecMap](https://github.com/artetxem/vecmap)) and then run:

```
python3 vecmap/map_embeddings.py --src_input ./data/mono/txt/en/WP/en.train.wp.vec --trg_input ./data/mono/txt/el/WP/el.train.wp.vec --src_output ./data/mono/txt/en/WP/mapped_en_el_embs.txt --trg_input ./data/mono/txt/el/WP/mapped_el_embs.txt
```

## Acknowledgements

We would like to thank the community for releasing their code! This repository contains code from [HuggingFace](https://github.com/huggingface/transformers) and from the [RAMEN](https://github.com/alexa/ramen) repository.

---
