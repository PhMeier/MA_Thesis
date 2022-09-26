import os

import wandb
from transformers import AutoTokenizer, pipeline, Trainer, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup
import datasets
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import pipeline, TrainingArguments, BertTokenizer
#import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
import transformers
CUDA_LAUNCH_BLOCKING=1

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['<t>'], special_tokens=True)
tokenizer.add_tokens(['</t>'], special_tokens=True)

save_directories = {"cl": "/workspace/students/meier/MA/BERT_mnli_filtered", "bw":"/pfs/work7/workspace/scratch/hd_rk435-checkpointz/amrbart_mnli_verid_text"}


#"""
#os.environ["WANDB_DIR"] = os.getcwd()
#os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
#wandb.login()
#wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")


def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")

def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #print("Preds: \n", preds)
    #preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=1) #-1)



if __name__ == "__main__":
    platform = ""

    paths = {"train_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_train_with_tags.csv",
             "test_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.csv",
             "train_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train_with_tags.csv",
             "test_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.csv",
             "train": "../../data/MNLI_filtered/MNLI_filtered/new_train.csv",
             "test": "../../data/MNLI_filtered/MNLI_filtered/new_dev_matched.csv"}
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    num_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2, "entailment\n": 0, "neutral\n": 1,
                    "contradiction\n": 2}
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.resize_token_embeddings(len(tokenizer))
    df_train = pd.read_csv(paths["train" + platform])
    df_val = pd.read_csv(paths["test" + platform])

    df_train["gold_label"] = df_train["gold_label"].map(num_to_label)
    df_train["gold_label"] = df_train["gold_label"].astype(int)
    df_train["sentence1"] = df_train["sentence1"].astype(str)
    df_train["sentence2"] = df_train["sentence2"].astype(str)

    df_val["gold_label"] = df_val["gold_label"].map(num_to_label)
    df_val["gold_label"] = df_val["gold_label"].astype(int)
    df_val["sentence1"] = df_val["sentence1"].astype(str)
    df_val["sentence2"] = df_val["sentence2"].astype(str)


    dataset_train_split = Dataset.from_pandas(df_train)
    dataset_val_split = Dataset.from_pandas(df_val)
    dataset_train_split = dataset_train_split.rename_column("sentence1", "premise")
    dataset_train_split = dataset_train_split.rename_column("sentence2", "hypothesis")
    dataset_train_split = dataset_train_split.rename_column("gold_label", "label")
    dataset_val_split = dataset_val_split.rename_column("sentence1", "premise")
    dataset_val_split = dataset_val_split.rename_column("sentence2", "hypothesis")
    dataset_val_split = dataset_val_split.rename_column("gold_label", "label")

    dataset_train_split = dataset_train_split.map(encode, batched=True)
    dataset_val_split = dataset_val_split.map(encode, batched=True)

    from transformers import TrainingArguments, Trainer

    #learning_rate = 5e-5
    optim = transformers.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = get_linear_schedule_with_warmup(optim,num_warmup_steps=1858, num_training_steps=30350)

#output_dir=save_directories[platform]
    training_args = TrainingArguments(evaluation_strategy="epoch", per_device_train_batch_size=16,
                                      logging_steps=50, per_device_eval_batch_size=8,
                                      eval_accumulation_steps=10, num_train_epochs=10, report_to="none",
                                      output_dir="./", gradient_checkpointing=False, fp16=False,
                                      save_total_limit=10, load_best_model_at_end=True,
                                      save_strategy="epoch", seed=42)  # disable wandb
    # preprocess_logits_for_metrics=preprocess_logits,
    # compute_metrics=compute_metrics

    # LR: 2e-05: correct
    # train batch correct, but no gradient accumulation steps
    # eval batch size set to 8

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_split,
        eval_dataset=dataset_val_split,
        compute_metrics=compute_metrics)

    trainer.train()
