import pandas as pd
import numpy as np
import multiprocessing
from easydict import EasyDict
import yaml
from datasets import load_dataset, load_metric

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import argparse
import torch
import torch.nn as nn
import torch.nn.utils as utils


def train(CFG):
    print("@LOG train(CFG)", CFG)
    metric = load_metric("sacrebleu")
    print("@LOG metric done")
    
    dset = load_dataset("csv", data_files={'train':CFG.train_data,
                                       'test': CFG.test_data})
    
    print("@LOG model name:", CFG.model_name) #QuoQA-NLP/KE-T5-En2Ko-Base
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, local_files_only=True)
    
    
def main():
    print('@LOG == main() ==')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation.', fromfile_prefix_chars='@')
    parser.add_argument('--mode', type=str, help='train or test', default='test')
    parser.add_argument('--yaml', type=str, help='yaml file', default='config.yaml')
    args = parser.parse_args()
    print("@LOG args", args)

    if args.yaml:
        with open(args.yaml) as infile:
            SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
            CFG = EasyDict(SAVED_CFG["CFG"])
            print("@LOG CFG ", CFG)

    if args.mode == "train":
        train(CFG)
    elif args.mode == "test":
        test(CFG)


        
        




#tokenization...
tokenized_datasets = dset.map(preprocess_function, batched=True, num_proc=multiprocessing.cpu_count())
print("[LOG] tokenized_datasets done")

#load pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name, local_files_only=True)
print("[LOG] AutoModelForSeq2SeqLM done")

#for logging
str_model_name = CFG.model_name.split("/")[-1]
run_name = f"{str_model_name}-finetuned-{CFG.src_language}-to-{CFG.tgt_language}"
print("[LOG] run_name", run_name)

training_args = Seq2SeqTrainingArguments(
    run_name,
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.valid_batch_size,
    evaluation_strategy=CFG.evaluation_strategy,
    # eval_steps=CFG.eval_steps,
    save_steps=CFG.save_steps,
    num_train_epochs=CFG.num_epochs,
    save_total_limit=CFG.num_checkpoints,
    predict_with_generate=True,
    fp16=CFG.fp16,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    logging_steps=CFG.logging_steps,
)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("[LOG YJ] Trainer Ready DONE!")

trainer.train()
trainer.evaluate()
trainer.save_model(CFG.save_path)

print("[LOG YJ] Trainer COMPLETE ALL JOBS HAVE DONE!")
