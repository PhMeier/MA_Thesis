"""
Inference script to test the trained models on the dev set.

"""

import evaluate
import torch
from datasets import load_dataset, load_metric
from evaluate import evaluator
from transformers import AutoTokenizer, pipeline, Trainer
from torch.utils.data import DataLoader
import datasets
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import pipeline, TrainingArguments
# import numpy as np
# np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys

CUDA_LAUNCH_BLOCKING = 1
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")


def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=1)


def add_tag_premise(s):
    # s = "<t> " + s
    s["premise"] = "<t> " + s["premise"]
    return s


def add_tag_hypothesis(s):
    s["hypothesis"] = s["hypothesis"] + " </t>"
    return s


if __name__ == "__main__":

    outputfile = sys.argv[1]
    eval_model = sys.argv[2]  # path to the model


    # /workspace/students/meier/MA/SOTA_Bart/best
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(eval_model, local_files_only=True)
    dataset_test_split = load_dataset("glue", "mnli", split='validation_matched')
    tokenizer.add_tokens(['<t>'], special_tokens=True)
    tokenizer.add_tokens(['</t>'], special_tokens=True)

    # add the tags
    updated_val = dataset_test_split.map(add_tag_premise)
    updated_val = dataset_test_split.map(add_tag_hypothesis)


    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, preprocess_logits_for_metrics=preprocess_logits,
                      compute_metrics=compute_metrics)
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test["validation_matched"])

    # res = trainer.predict(dataset_test_split["test"])

    print(res)
    print(res.label_ids)
    # print(res.label_ids.reshape(107, 14).tolist())
    pd.DataFrame(res.predictions).to_csv("/home/students/meier/MA/results/mnli/val" + outputfile,
                                         header=["label"])  # "results_mnli_matched_bartLarge.csv")
    print(res.metrics)
