
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
#import numpy as np
#np.set_printoptions(threshold=np.inf)
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
    #preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result

def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=1)



if __name__ == "__main__":
    paths = {"cl_data_pos": "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_nor.csv",
             "cl_data_neg": "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_neg.csv", #full_verb_veridicality.csv",
             "cl_model": "/workspace/students/meier/MA/Bart_verid/text/amrbart_mnli_filtered_text_17/checkpoint-2277", #"/workspace/students/meier/MA/amrbart_mnli_verid_text/checkpoint-3036", #BART_veridicality_text/checkpoint-15175/", #"/workspace/students/meier/MA/SOTA_Bart/best/checkpoint-12000/",
             "tow_model": "../checkpoint-12000/",
             "cl_model_verid_text": "/workspace/students/meier/MA/amrbart_mnli_verid_text/checkpoint-3036",
             "cl_model_67": "/workspace/students/meier/MA/Bart_verid/text/amrbart_mnli_filtered_text_67/checkpoint-5313",      
             "tow_data": "C:/Users/Meier/Projekte/MA_Thesis/preprocess/verb_verid_neg.csv",
             "cl_model_graph": "/workspace/students/meier/MA/Bart_verid/graph/amrbart_mnli_filtered_only_graph_67/checkpoint-3795", #"/workspace/students/meier/MA/amrbart_mnli_filtered_only_graph/checkpoint-2277",
             "cl_data_graph_pos": "/home/students/meier/MA/MA_Thesis/preprocess/veridicality_positive_test_graph.csv",
             "cl_data_graph_neg": "/home/students/meier/MA/MA_Thesis/preprocess/veridicality_negated_test_graph.csv",
             "cl_joint_pos": "/home/students/meier/MA/MA_Thesis/preprocess/veridicality_positive_test_joint_input.csv",
             "cl_joint_neg": "/home/students/meier/MA/MA_Thesis/preprocess/veridicality_negated_test_joint_input.csv",
             "bart_17": "/workspace/students/meier/MA/Bart_verid/text/bart_17/checkpoint-2277",
             "cl_text_tags_pos": "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_nor_with_tags.csv",
             "cl_text_tags_neg": "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_neg_with_tags.csv"}
    suffix = sys.argv[1]
    outputfile = sys.argv[2]
    eval_model = sys.argv[3]
    data_part = sys.argv[4]
    
    # /workspace/students/meier/MA/SOTA_Bart/best
    path = "../checkpoint-12000/"  # "../checkpoint-12000/"
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(eval_model, local_files_only=True)
    dataset_test_split = load_dataset("csv", data_files={"test": paths[data_part]})
    #dataset_test_split = load_dataset("glue", "mnli", split='test_matched')
    #dataset_test_split = dataset_test_split.remove_columns("label")
    #tokenized_datasets_test = dataset_test_split.rename_column("signature", "label")
    #tokenized_datasets_test = tokenized_datasets_test.rename_column("sentence", "premise")
    #tokenized_datasets_test = tokenized_datasets_test.rename_column("complement", "hypothesis")
    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, preprocess_logits_for_metrics=preprocess_logits, compute_metrics=compute_metrics)
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test["test"])

    #res = trainer.predict(dataset_test_split["test"])

    print(res)
    print(res.label_ids)
    #print(res.label_ids.reshape(107, 14).tolist())
    pd.DataFrame(res.predictions).to_csv("/home/students/meier/MA/results/"+outputfile, header=["label"]) #"results_mnli_matched_bartLarge.csv")
    print(res.metrics)

