# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "False"
# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import json
import numpy as np
import glob
import torch
from tqdm import tqdm
from datasets import load_dataset

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    set_seed, BertTokenizerFast, BertForMaskedLM, WEIGHTS_NAME,
)
from transformers.trainer_utils import is_main_process

from utils.models import ramen, AlignAnchor
from utils.trainer import Trainer, BiTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    foreign_model: Optional[str] = field(
        default='emb_layer/el/bert-el_ours_align_embs',
        metadata={
            "help": "The model checkpoint for weights initialization for the foreign model."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    biLM_model_name: Optional[str] = field(
        default="ours",
        metadata={"help": "The name of the biLM model"},
    )
    share_position_embs: Optional[bool] = field(
        default=True,
        metadata={"help": "If positional embeddings are shared among source and target languages"},
    )
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train the whole model or only the embedding layer"},
    )
    alignment_dir: Optional[str] = field(
        default="alignments/en-el/",
        metadata={"help": "The path to the alignment dict"},
    )
    model_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tgt_tokenizer_name: Optional[str] = field(
        default='alignments/en-el/new_tgt_vocab.txt',
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    do_lower_case: bool = field(
        default=True, metadata={"help": "Set this flag if you are using an uncased model."}
    )
    cache_dir: Optional[str] = field(
        default='cache',
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tgt_lang: str = field(default='el', metadata={"help": "The language that we will transfer the LM to ."})

    src_train_file: str = field(default='data/mono/txt/en/en.train.txt',
                                metadata={"help": "The input training data file (a text file)."})

    src_validation_file: str = field(
        default='data/mono/txt/en/en.valid.txt',
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
            and not model_args.model_ckpt
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # save model and data arguments
    json_dict = vars(model_args)
    json_dict.update(vars(data_args))
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(training_args.output_dir + "/training_arguments.json", 'w') as f:
        json.dump(json_dict, f, indent=2)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    src_data_files = {}
    src_data_files["train"] = data_args.src_train_file
    src_data_files["validation"] = data_args.src_validation_file
    extension = data_args.src_train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    src_datasets = load_dataset(extension, data_files=src_data_files)

    tgt_data_files = {}
    tgt_data_files["train"] = 'data/mono/txt/{}/{}.train.txt'.format(data_args.tgt_lang,
                                                                                            data_args.tgt_lang)
    tgt_data_files["validation"] = 'data/mono/txt/{}/{}.valid.txt'.format(
        data_args.tgt_lang,
        data_args.tgt_lang)
    extension = tgt_data_files["train"].split(".")[-1]
    if extension == "txt":
        extension = "text"
    tgt_datasets = load_dataset(extension, data_files=tgt_data_files)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    src_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
        do_lower_case=model_args.do_lower_case,
    )
    tgt_tokenizer = BertTokenizerFast(
        vocab_file=model_args.tgt_tokenizer_name, do_lower_case=model_args.do_lower_case, strip_accents=False)

    src_model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    tgt_model = AutoModelForMaskedLM.from_pretrained(
        model_args.foreign_model,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if model_args.biLM_model_name == "ramen":
        model = ramen(src_model, tgt_model, share_position_embs=model_args.share_position_embs)

        if model_args.freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.src_model.base_model.embeddings.parameters():
                param.requires_grad = True
            for param in model.tgt_model.base_model.embeddings.parameters():
                param.requires_grad = True
    elif model_args.biLM_model_name in ["ours", "joint"]:
        if not model_args.alignment_dir:
            raise ValueError("No alignment directory provided")

        with open(os.path.join(model_args.alignment_dir, "src_shared_idx.json"), 'r') as f:
            src_subw_idx = json.load(f)

        logger.info("  Number of shared subwords {}".format(len(src_subw_idx)))

        non_shared_mask = np.ones(src_tokenizer.vocab_size, dtype=bool)
        non_shared_mask[src_subw_idx] = False
        non_shared_src_idx = np.arange(tgt_tokenizer.vocab_size)[non_shared_mask]

        model = AlignAnchor(src_model, tgt_model, non_shared_src_idx=non_shared_src_idx,
                            share_position_embs=model_args.share_position_embs)

        if model_args.freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.src_model.base_model.embeddings.parameters():
                param.requires_grad = True
            for param in model.tgt_model.base_model.embeddings.parameters():
                param.requires_grad = True
            model.non_shared_tgt_subw.requires_grad = True
            model.src_embs.requires_grad = True
    else:
        raise ValueError("The model name is not correct")

    if (training_args.do_eval and not training_args.do_train) or model_args.model_ckpt is not None:
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(training_args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        checkpoint = checkpoints[np.argmax(
            [int(ckpt.split("/")[-1].split("-")[-1]) for ckpt in checkpoints if "checkpoint" in ckpt])]
        model.load_state_dict(torch.load(checkpoint + "/" + WEIGHTS_NAME))
        logger.info("  Loading from checkpoint {}".format(checkpoint))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = src_datasets["train"].column_names
    else:
        column_names = src_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    logger.info("  Tokenizing data using  {}".format(model_args.tgt_tokenizer_name))

    tokenized_datasets = []
    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        for tokenizer, datasets, lang in zip([src_tokenizer, tgt_tokenizer], [src_datasets, tgt_datasets],
                                             ['src', data_args.tgt_lang]):
            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=data_args.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            dataset_filenames = {'train': 'data/{}_{}_train_file'.format(model_args.biLM_model_name, lang),
                                 'validation': 'data/{}_{}_val_file'.format(model_args.biLM_model_name, lang)}
            tokenized_datasets.append(datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_names=dataset_filenames
            ))
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        for tokenizer, datasets, lang in zip([src_tokenizer, tgt_tokenizer], [src_datasets, tgt_datasets],
                                             ['src', data_args.tgt_lang]):
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            dataset_filenames = {'train': 'data/{}_{}_train_file'.format(model_args.biLM_model_name, lang),
                                 'validation': 'data/{}_{}_val_file'.format(model_args.biLM_model_name, lang)}
            tokenized_dataset = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_names=dataset_filenames
            )

            if data_args.max_seq_length is None:
                max_seq_length = tokenizer.model_max_length
            else:
                if data_args.max_seq_length > tokenizer.model_max_length:
                    logger.warn(
                        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            dataset_filenames = {'train': 'data/{}_{}_train_file'.format(model_args.biLM_model_name, lang),
                                 'validation': 'data/{}_{}_val_file'.format(model_args.biLM_model_name, lang)}
            tokenized_datasets.append(tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_names=dataset_filenames
            ))

    # Data collator
    # This one will take care of randomly masking the tokens.
    src_data_collator = DataCollatorForLanguageModeling(tokenizer=src_tokenizer,
                                                        mlm_probability=data_args.mlm_probability)
    tgt_data_collator = DataCollatorForLanguageModeling(tokenizer=tgt_tokenizer,
                                                        mlm_probability=data_args.mlm_probability)

    training_args.per_device_train_batch_size = int(training_args.per_device_train_batch_size / 2)

    # Initialize our Trainer
    trainer = BiTrainer(
        model=model,
        args=training_args,
        src_train_dataset=tokenized_datasets[0]["train"] if training_args.do_train else None,
        tgt_train_dataset=tokenized_datasets[1]["train"] if training_args.do_train else None,
        src_eval_dataset=tokenized_datasets[0]["validation"] if training_args.do_eval else None,
        tgt_eval_dataset=tokenized_datasets[1]["validation"] if training_args.do_eval else None,
        src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer,
        src_data_collator=src_data_collator, tgt_data_collator=tgt_data_collator,
    )

    print('Number of trainable parameters :{}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad == True)))
    # Training
    if training_args.do_train:
        if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)):
            model_path = model_args.model_name_or_path
        elif model_args.model_ckpt is not None:
            model_path = model_args.model_ckpt
        else:
            model_path = None
        trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        print(eval_output["eval_loss"])
        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


if __name__ == "__main__":
    main()
