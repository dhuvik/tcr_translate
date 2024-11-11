#########################################################
############## Script for Fine-Tuning TCRT5 #############
#########################################################

### Loading Dependencies
# Import Standard Libraries
import os
import yaml
import argparse

# Import Third-Party Libraries
import numpy as np
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from accelerate import Accelerator

# Custom Modules
from src.tokenizer import TCRT5Tokenizer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune TCRT5 model.')
parser.add_argument(
    '--config',
    type=str,
    default='config.yaml',
    help='Path to the configuration YAML file.'
)
args = parser.parse_args()

# Get current and parent directory paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)

# Load configuration from the specified config file
config_path = os.path.join(curr_dir, args.config_file)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

### Load Datasets
# Load source and target files separately
source_dataset = load_dataset(
    'text',
    data_files=os.path.join(parent_dir, config['data']['train_source'])
)
target_dataset = load_dataset(
    'text',
    data_files=os.path.join(parent_dir, config['data']['train_target'])
)

# Load the validation files
val_source = load_dataset(
    'text',
    data_files=os.path.join(parent_dir, config['data']['val_source'])
)
val_target = load_dataset(
    'text',
    data_files=os.path.join(parent_dir, config['data']['val_target'])
)

# Ensure source and target datasets have the same size
assert len(source_dataset["train"]) == len(target_dataset["train"]), "Training source and target datasets have different lengths."
assert len(val_source['train']) == len(val_target['train']), "Validation source and target datasets have different lengths."

# Merge source and target datasets
train_dataset = Dataset.from_dict({
    'src_texts': [example['text'] for example in source_dataset['train']],
    'tgt_texts': [example['text'] for example in target_dataset['train']]
})

val_dataset = Dataset.from_dict({
    'src_texts': [example['text'] for example in val_source['train']],
    'tgt_texts': [example['text'] for example in val_target['train']]
})

### Tokenization
# Create tokenizer
tokenizer = TCRT5Tokenizer().from_pretrained(config['tokenizer']['name'])

def labeled_tokenize_function(example, src_max_len, trg_max_len):
    # Split each element of the list into two parts
    sentences = [text.split(" ", 1) for text in example["src_texts"]]
    # Create lists for source and target texts
    pep = [s[0] for s in sentences]
    pseudo = [s[1] if len(s) > 1 else "" for s in sentences]  # Ensure that the second part exists
    src_sequences = list(zip(pep, pseudo))
    target_text = example["tgt_texts"]

    # Tokenize source and target texts separately
    source_tokens = tokenizer.tokenize_pmhc(
        src_sequences,
        padding="max_length",
        return_tensors='pt',
        truncation=True,
        max_length=src_max_len
    )
    target_tokens = tokenizer.tokenize_tcr(
        target_text,
        padding="max_length",
        return_tensors='pt',
        truncation=True,
        max_length=trg_max_len
    )

    # Ensure correct lengths are taken for source and target sequences and their attention masks
    padded_source = {
        "input_ids": source_tokens["input_ids"][:, :src_max_len],
        "attention_mask": source_tokens["attention_mask"][:, :src_max_len]
    }
    padded_target = {
        "input_ids": target_tokens["input_ids"][:, :trg_max_len],
        "attention_mask": target_tokens["attention_mask"][:, :trg_max_len]
    }

    return {
        "input_ids": padded_source["input_ids"],
        "attention_mask": padded_source["attention_mask"],
        "labels": padded_target["input_ids"]
    }

# Tokenize datasets
tokenized_dataset = train_dataset.map(
    lambda x: labeled_tokenize_function(
        x,
        src_max_len=config['tokenizer']['src_max_len'],
        trg_max_len=config['tokenizer']['trg_max_len']
    ),
    batched=True
)
tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

tokenized_val = val_dataset.map(
    lambda x: labeled_tokenize_function(
        x,
        src_max_len=config['tokenizer']['src_max_len'],
        trg_max_len=config['tokenizer']['trg_max_len']
    ),
    batched=True
)
tokenized_val.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

### Model and Training Setup
# Instantiate the Pre-Trained Model
model = T5ForConditionalGeneration.from_pretrained(config['model']['name'])

# Make the training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(parent_dir, config['training_args']['output_dir']),
    overwrite_output_dir=True,
    num_train_epochs=config['training_args']['num_train_epochs'],
    per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
    per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],
    eval_steps=config['training_args']['eval_steps'],
    save_steps=config['training_args']['save_steps'],
    do_eval=True,
    evaluation_strategy='steps',
    learning_rate=config['training_args']['learning_rate'],
    weight_decay=config['training_args']['weight_decay'],
    logging_steps=config['training_args']['logging_steps'],
    save_total_limit=config['training_args']['save_total_limit'],
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# Prepare for training with Accelerator
accelerator = Accelerator()
trainer = accelerator.prepare(trainer)

### Training
trainer.train()

