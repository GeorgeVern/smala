# *SMALA* - *S*ubword *M*apping and *A*nchoring across *La*nguages
This repository contains source code for our EMNLP 2021 Findings paper: Subword Mapping and Anchoring Across Languages.

## Introduction
In this work we propose a novel method to construct bilingual subword vocabularies. We identify _false positives_ (identical subwords with different meanings across languages) and _false negatives_ (different subwords with similar meanings) as limitation of jointly constructed subword vocabularies. SMALA extracts subword alignments using an unsupervised state-of-the-art mapping technique and uses them to create cross-lingual anchors based on subword similarities.

## Model
We first learn **subwords** separately for each language and then train the corresponding embeddings. We then apply a **mapping** method to obtain similarity scores between the embeddings, which we use to extract **alignments** between subwords of the two languages. We finally tie the parameters of the aligned subwords to create **anchors** during training. 

SMALA  outperforms current methods for joint construction of multilingual subword vocabulariesin cases where there is no cross-lingual signal, such as zero-shot transfer to an unseen language (XNLI) only by sharing subword embeddings. When cross-lingual supervision is available, SMALA is a viable alternative to create shared bilingual vocabularies.

## Prerequisites
### Dependencies
* Python 3.7.9
* [Pytorch](https://pytorch.org/) (tested on 1.6.0)
* [FastText](https://github.com/facebookresearch/fastText)
* [FastAlign](https://github.com/clab/fast_align)
* [Transformers](https://huggingface.co/transformers/) (tested on 1.4.0)
* [Tokenizers](https://github.com/huggingface/tokenizers) (tested on 0.9.4)


### Install Requirements
*Create Environment (Optional):* Ideally, you should create an environment for the project.

    conda create -n smala_env python=3.7.9
    conda activate smala_env
Install PyTorch `1.1.0` with the desired Cuda version if you want to use the GPU:

`conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch`

Clone the project:

```
git clone https://github.com/GeorgeVern/smala.git
cd smala
```

Then install the rest of the requirements:

`pip install -r requirements.txt`

### Download Data



## Acknowledgements

We would like to thank the community for releasing their code! This repository contains code from [HuggingFace](https://github.com/huggingface/transformers) and from the [RAMEN](https://github.com/alexa/ramen) repository.

---
