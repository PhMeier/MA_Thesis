import evaluate
import torch
from transformers import AutoTokenizer, pipeline, Trainer
import datasets
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import pipeline, TrainingArguments
from nltk.tokenize import sent_tokenize
#import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
CUDA_LAUNCH_BLOCKING=1
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

meteor_score = evaluate.load("meteor")
def encode(examples):
    model_inputs = tokenizer(examples["premise"], truncation=True, padding='max_length', max_length=1024)
    labels = tokenizer(examples['hypothesis'], truncation=True, padding='max_length', max_length=150)
    model_inputs["labels"] = labels["input_ids"] #model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"] # model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


def compute_metrics(pred):  # eval_pred):
    res = {}
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]
    decoded_predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids==-100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_predictions]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    print("Decoded preds: ", decoded_preds)
    print("Decoded Labels: ", decoded_labels)
    score_meteor = meteor_score.compute(predictions=decoded_preds, references=decoded_labels)
    print(score_meteor)
    #result = {key: value.mid.fmeasure * 100 for key, value in score_rouge.items()}
    #rounded_res = {k: round(v,4) for k,v in result.items()}
    rounded_res = round(score_meteor['meteor'], 2) #/len(results['f1']), 2)
    res["meteor"]= rounded_res
    return res


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



if __name__ == "__main__":
    paths = {"cl_kaggle_data": "/home/students/meier/MA/multinli_0.9_test_matched_unlabeled_mod.csv", # prodvided dataset from webside
             "cl_kaggle_data_graph": "/home/students/meier/MA/MA_Thesis/preprocess/MNLI_test_set_kaggle_graph.csv",
             "cl_kaggle_data_joint": "/home/students/meier/MA/MA_Thesis/preprocess/MNLI_test_set_kaggle_joint.csv",
             "cl_kaggle_data_text": "/home/students/meier/MA/multinli_0.9_test_matched_unlabeled_mod_with_tags_corrected.csv"}
    model_path = sys.argv[1]
    data_type = sys.argv[2]
    outputfile = sys.argv[3]


    new_index = [i for i in range(9847, 19643)] # This index is needed for the kaggle data
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    # /workspace/students/meier/MA/SOTA_Bart/best
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    df = pd.read_csv(paths["cl_kaggle_data_text"], delimiter=",")
    #dataset_test_split = load_dataset("csv", data_files={"test": paths["cl_kaggle_data"]})
    #dataset_test_split = load_dataset("glue", "mnli", split='test_matched')
    tokenized_datasets_test = Dataset.from_pandas(df)
    #dataset_test_split = dataset_test_split.remove_columns("label")
    #tokenized_datasets_test = dataset_test_split.rename_column("signature", "label")
    #tokenized_datasets_test = dataset_test_split.rename_column("sentence1", "premise")
    #tokenized_datasets_test = tokenized_datasets_test.rename_column("sentence2", "hypothesis")
    tokenized_datasets_test = tokenized_datasets_test.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, preprocess_logits_for_metrics=preprocess_logits) #compute_metrics=compute_metrics
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test) #["test"])
    print(res)
    print(res.predictions)
    #print(res.label_ids.reshape(107, 14).tolist())
    final_dataframe = pd.DataFrame({"pairID": new_index, "gold_label": res.predictions})
    final_dataframe.to_csv(outputfile, index=False, header=["pairID", "gold_label"])
    #final_dataframe.DataFrame(res.predictions).to_csv("results_mnli_matched_kaggle_bartLarge.csv")
    #print(res.metrics)
