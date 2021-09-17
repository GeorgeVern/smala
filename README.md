# *SMALA* - *S*ubword *M*apping and *A*nchoring across *La*nguages
This repository contains source code for our EMNLP 2021 Findings paper: [Subword Mapping and Anchoring across Languages](https://arxiv.org/abs/2109.04556v1).

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
* [FastAlign](https://github.com/clab/fast_align) (requires [cmake](https://cmake.org/install/))
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
Download and preprocess wikipedia data and learn language-specific for **English** (en) and another language, e.g. **Greek** (el):
    
    bash get-mono-data.sh en
    bash get-mono-data.sh el el-tokenizer
    
### 1) Subword Mapping
Learn subword embeddings for each language:
    
    bash learn_subw_embs.sh en
    bash learn_subw_embs.sh el el-tokenizer

Map the monolingual subword embeddings into a common space using the **unsupervised** version of  VecMap, since we don't want to rely on seed dictionaries or identical (sub)words. Clone the github repo of ([VecMap](https://github.com/artetxem/vecmap)) and then run:

```
python3 vecmap/map_embeddings.py --unsupervised smala/data/mono/txt/en/WP/en.train.wp.vec smala/data/mono/txt/el/WP/el.train.wp.vec smala/data/mono/txt/en/WP/mapped_en_el_embs.txt smala/data/mono/txt/el/WP/mapped_el_embs.txt
```
### 2) Anchoring of similar subwords
Extract subword alignments from the mapped subword embeddings:

    python3 extract_alignments.py --src_emb data/mono/txt/en/WP/mapped_en_el_embs.txt --tgt_emb  data/mono/txt/el/WP/mapped_el_embs.txt --similarity cosine --alignment_dir en-el --initialize
    
Create new vocabulary for the target language (so that aligned subwords point to the same embedding in both langauges) based on the alignments:

    python3 utils/create_new_vocabs.py --tgt_tokenizer el-tokenizer --model_type ours --alignment_dir alignments/en-el
    
Initialize the embedding layer of the target model:

    python3 utils/init_weight.py --tgt_vocab alignments/en-el/new_tgt_vocab.txt --prob alignments/en-el/prob_vector --tgt_model emb_layer/el/bert-ours_align_embs

### Compare with other models
The above steps serve to employ SMALA with additional initialization of the non-aligned subwords (`ours+align` in the paper). To compare with the other models that are included in the paper you need to modify these steps:
* `ours`: as above but run the `extract_alignments.py` script without the flag ~~`--initialize~`~~ and the `init_weight.py` script with the `--prob None` flag. 
* `joint`: skip the subword mapping and the first step of anchoring, run the `extract_alignments.py` script with the `--similarity surface_form` and without the  ~~`--initialize~`~~ flag, run the `create_new_vocabs.py` script with the `--model_type joint` flag and the `init_weight.py` script with the `--prob None` flag.
* `ramen`: skip the above steps, see [RAMEN](https://github.com/alexa/ramen) on how to create the probabilty vector (we also lowercase) and run the `init_weight.py` script with the correct`--prob` flag and the original tokenizer (e.g. `--tgt_vocab el-tokenizer/vocab.txt`)

### Extract alignments from parallel data
Our method can also exploit parallel data (in the paper we use data from [Europarl](https://www.statmt.org/europarl/) and [United Nations](https://conferences.unite.un.org/uncorpus)). To do so you must first download (e.g. in `data/para/en-el`) and preprocess (tokenize and lowercase) a parallel corpus. Then run:

    python3 utils/apply_tokenizer.py --tokenizer bert --file data/para/en-el/en-el.en.txt
    python3 utils/apply_tokenizer.py --tokenizer el-tokenizer --file data/para/en-el/en-el.el.txt

Then run FastAlign:
    
    bash run_fast-align.sh en el data/para/en-el/WP/en-el.en.wp data/para/en-el/WP/en-el.el.wp data/para/en-el/WP/fast-align
    
To get the similarity matrix from fast-align output clone the [RAMEN](https://github.com/alexa/ramen) repo and run:

    python3 ramen/code/alignment/get_prob_para.py --bitxt smala/data/para/en-el/WP/fast-align/cleared.en-el --align smala/data/para/en-el/WP/fast-align/align.en-el --save smala/data/para/en-el/WP/fast-align/probs.para.en-el.pth
    
Finally, to extract alignments, create new vocabulary and initialize the embedding layer of the target model, run:

    python3 extract_alignments_para.py --tgt_tokenizer el-tokenizer --similarity_matrix data/para/en-el/WP/fast-align/probs.para.en-el.pth --alignment_dir en-el_fastalign
    python3 utils/create_new_vocabs.py --tgt_tokenizer el-tokenizer --model_type ours --alignment_dir alignments/en-el_fastalign
    python3 utils/init_weight.py --tgt_vocab alignments/en-el_fastalign/new_tgt_vocab.txt --prob alignments/en-el_fastalign/prob_vector --tgt_model emb_layer/el/bert-ours_align_para_embs



## Language Model Transfer with SMALA
To transfer a pretrained LM to a new language using SMALA run:

    python3 fine-tune_biBERTLM.py \
    --tgt_lang el \
    --output_dir ckpts/greek_ours_align \
    --foreign_model emb_layer/el/bert-ours_align_embs \
    --biLM_model_name ours \
    --alignment_dir alignments/en-el \
    --tgt_tokenizer_name alignments/en-el/new_tgt_vocab.txt \
    --do_train --do_eval \
    --evaluation_strategy steps \
    --seed 12 \
    --per_device_eval_batch_size 38 \
    --max_steps 120000 \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --per_device_train_batch_size 38 \
    --eval_accumulation_steps 1


## Acknowledgments

We would like to thank the community for releasing their code! This repository contains code from [HuggingFace](https://github.com/huggingface/transformers) and from the [RAMEN](https://github.com/alexa/ramen), [VecMap](https://github.com/artetxem/vecmap), [XLM](https://github.com/facebookresearch/XLM) and [SimAlign](https://github.com/cisnlp/simalign) repositories.

---
## Reference
If you use this repo in your research, please cite the paper:

    @misc{vernikos2021subword,
        title={Subword Mapping and Anchoring across Languages},
        author={Giorgos Vernikos and Andrei Popescu-Belis},
        year={2021},
        eprint={2109.04556},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
