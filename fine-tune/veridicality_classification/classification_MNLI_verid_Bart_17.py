import os
import wandb
from transformers import AutoTokenizer, pipeline, Trainer
import datasets
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import pipeline, TrainingArguments
# import numpy as np
# np.set_printoptions(threshold=np.inf)
import pandas as pd
import transformers

save_directories = {"cl": "/workspace/students/meier/MA/Bart_verid", "bw":"/pfs/work7/workspace/scratch/hd_rk435-checkpointz/amrbart_mnli_verid_17"}
CUDA_LAUNCH_BLOCKING = 1
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# add special tokens

#save_directories = {"cl": "/workspace/students/meier/MA/BART_veridicality_text",
#                    "bw": "/pfs/work7/workspace/scratch/hd_rk435-checkpointz/amrbart_mnli_verid"}

# """
os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
# wandb.login()
wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")


def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    print("Preds: \n", preds)
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
    platform = "bw"

    paths = {"train_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_train_no_tags.csv",
             "test_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_dev_matched_no_tags.csv",
             "train_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train_with_tags.tsv",
             "test_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.tsv",
             "train": "../data/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "test": "../data/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"}
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    num_to_label = {"entailment": 0, "neutral":1, "contradiction" :2}
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    num_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    model = BartForSequenceClassification.from_pretrained("facebook/bart-large")
    model.resize_token_embeddings(len(tokenizer))
    df_train = pd.read_csv(paths["train_data_"+platform])
    df_val = pd.read_csv(paths["test_data_"+platform])

    df_train["gold_label"] = df_train["gold_label"].map(num_to_label)
    df_train["gold_label"] = df_train["gold_label"].astype(int)
    df_train["sentence1"] = df_train["sentence1"].astype(str)
    df_train["sentence2"] = df_train["sentence2"].astype(str)

    df_val["gold_label"] = df_val["gold_label"].map(num_to_label)
    df_val["gold_label"] = df_val["gold_label"].astype(int) 
    df_val["sentence1"] = df_val["sentence1"].astype(str)
    df_val["sentence2"] = df_val["sentence2"].astype(str)

    df_train = df_train.drop('label1', axis=1)
    df_val = df_val.drop('label1', axis=1)


    dataset_train_split = Dataset.from_pandas(df_train)
    dataset_val_split = Dataset.from_pandas(df_val)
    dataset_train_split = dataset_train_split.rename_column("sentence1", "premise")
    dataset_train_split = dataset_train_split.rename_column("sentence2", "hypothesis")
    dataset_train_split = dataset_train_split.rename_column("gold_label", "label")
    dataset_val_split = dataset_val_split.rename_column("sentence1", "premise")
    dataset_val_split = dataset_val_split.rename_column("sentence2", "hypothesis")
    dataset_val_split = dataset_val_split.rename_column("gold_label", "label")

    #dataset_train_split = dataset_train_split.select(range(10))
    #dataset_val_split = dataset_val_split.select(range(10))    

    dataset_train_split = dataset_train_split.map(encode, batched=True)
    dataset_val_split = dataset_val_split.map(encode, batched=True)



    from transformers import TrainingArguments, Trainer

    learning_rate = 5e-5
    optim = transformers.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)

    training_args = TrainingArguments(evaluation_strategy="epoch", per_device_train_batch_size=16,
                                      gradient_accumulation_steps=8, logging_steps=50, per_device_eval_batch_size=2,
                                      eval_accumulation_steps=10, num_train_epochs=10, report_to="wandb",
                                      output_dir=save_directories[platform], gradient_checkpointing=True, fp16=True,
                                      save_total_limit=10, load_best_model_at_end=True,
                                      save_strategy="epoch", seed=17)  # disable wandb
    # preprocess_logits_for_metrics=preprocess_logits,
    # compute_metrics=compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_split,
        eval_dataset=dataset_val_split,
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=compute_metrics,
        optimizers=(optim, transformers.get_polynomial_decay_schedule_with_warmup(optim,
                                                                                  num_warmup_steps=1858,
                                                                                  num_training_steps=30680)),

    )
    trainer.train()