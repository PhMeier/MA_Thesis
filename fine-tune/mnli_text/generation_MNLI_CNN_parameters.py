import transformers
from transformers import AutoTokenizer, BartForSequenceClassification, BartTokenizer
from transformers import BartForConditionalGeneration
from datasets import load_dataset
import datasets
import numpy as np
from datasets import load_metric
#from bleu import list_bleu
import evaluate
from nltk.tokenize import sent_tokenize
import os
import wandb
import torch
#import rouge_score


rouge_score = evaluate.load("rouge") #"rouge") #"/home/hd/hd_hd/hd_rk435/rouge/rouge.py")  # evaluate.load("rouge") #/pfs/data5/home/hd/hd_hd/hd_rk435/ma_venv/lib64/python3.8/site-packages/rouge_score/rouge_scorer.py") #"rouge")
# Tokenizerfunktion umbauen done
# Metrik umbauen
# prüfen obs was taugt



save_directories = {"cl": "/workspace/students/meier/MA/generation/SOTA_Bart_Generation_cnn_parameters",
                    "bw": "/pfs/work7/workspace/scratch/hd_rk435-checkpointz/generation/bart_mnli_rouge_4gpus"}

os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
#wandb.login()
wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")

dataset_train = load_dataset("glue", "mnli", split='train')  # , download_mode="force_redownload")
dataset_val = load_dataset("glue", "mnli", split='validation_matched')
# dataset = load_dataset("glue", "mnli")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
# model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large")
# model = BartForSequenceClassification.from_pretrained("xfbai/AMRBART-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
print("Model Loaded")



def encode(examples):
    model_inputs = tokenizer(examples["premise"], truncation=True, padding='max_length', max_length=1024)
    labels = tokenizer(examples['hypothesis'], truncation=True, padding='max_length', max_length=150)
    model_inputs["labels"] = labels["input_ids"] #model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"] # model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


#dataset_train = dataset_train.select(10)
#dataset_val = dataset_val.select(10)

tokenized_datasets_t = dataset_train.map(encode, batched=True)
tokenized_datasets_v = dataset_val.map(encode, batched=True)
# tokenized_datasets = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
small_train_dataset = tokenized_datasets_t.shuffle(seed=42).select(range(50000))
small_eval_dataset = tokenized_datasets_v.shuffle(seed=42).select(range(2500))
# small_test_dataset = tokenized_datasets["test_matched"].shuffle(seed=42).select(range(1000))
# print(type(small_train_dataset))

print(small_train_dataset[0])
print(small_eval_dataset[-1])

metric = load_metric("accuracy")


def compute_metrics(pred):  # eval_pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]
    decoded_predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids==-100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_predictions]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    print("Decoded preds: ", decoded_preds)
    print("Decoded Labels: ", decoded_labels)
    score_rouge = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    print(score_rouge)
    result = {key: value.mid.fmeasure * 100 for key, value in score_rouge.items()}
    rounded_res = {k: round(v,4) for k,v in result.items()}
    return rounded_res
    """
    # decoden, dann score mit bleu
    #print("\n p \n ", p)
    preds, labels = p
    #print("\nbefore")
    #print("preds: ", preds)
    #print("labels: ", labels)
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #print("After")
    #print(preds)
    decoded_predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print("Decoded labels: \n", decoded_labels)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_predictions]
    #print("Decoded_preds \n", decoded_preds)
    #print("Length decoded preds: ", len(decoded_preds))
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    #print("Decoded labels2 : \n", decoded_labels)
    #print(len(decoded_labels))
    #score = list_bleu(decoded_preds, decoded_labels)
    score_rouge = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #print(score):wq
    #print("Score rouge: ", score_rouge)
    # score_rouge does not have .mid
    #result = {key: value.mid.fmeasure * 100 for key, value in score_rouge.items()}

    #print("result: ", result)
    rounded_res = {k: round(v,4) for k,v in score_rouge.items()}
    #print("Rounded Res: ", rounded_res)
    return rounded_res
    #return score
    """
    """
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result
    """
# https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/12
def preprocess_logits(logits, labels):
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    print("logits: \n", logits)
    print("logits argmax \n", logits.argmax(dim=1))
    return logits.argmax(dim=1) #-1) changed to 1 for BARTforConditionalGeneration
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

from transformers import TrainingArguments, Trainer

learning_rate = 3e-5
num_epochs = 10
training_steps = num_epochs * len(small_train_dataset) #len(dataset_train)
warmup_steps = (training_steps/100)*10
optim = transformers.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

# set fp16 to tru
# train_batch_size 16
training_args = TrainingArguments(evaluation_strategy="epoch", per_device_train_batch_size=16,
                                  gradient_accumulation_steps=8, logging_steps=50, per_device_eval_batch_size=4,
                                  eval_accumulation_steps=8, num_train_epochs=num_epochs, report_to="wandb",
                                  output_dir=save_directories["cl"],
                                  gradient_checkpointing=True, fp16=True, save_strategy="epoch", save_total_limit=30,
                                  load_best_model_at_end=True)  # disable wandb

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
    optimizers=(optim, transformers.get_polynomial_decay_schedule_with_warmup(optim,
                                                                              num_warmup_steps=warmup_steps,
                                                                              num_training_steps=training_steps)),
)
trainer.train()
# print_summary(result)
