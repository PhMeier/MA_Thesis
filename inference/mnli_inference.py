"""
Previous inference_amrbart

Inference procedure for MNLI data

Example call:
python3 mnli_inference.py path/to/model output.csv joint
"""

from transformers import Trainer
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import TrainingArguments
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
    return logits.argmax(dim=-1)


if __name__ == "__main__":
    paths = {"cl_data": "/home/students/meier/MA/MA_Thesis/preprocess/test.csv",  # full_verb_veridicality.csv",
             "cl_model": "/workspace/students/meier/MA/AMR_Bart/checkpoint-6136",
             "cl_kaggle_data": "/home/students/meier/MA/multinli_0.9_test_matched_unlabeled_mod.csv",
             # prodvided dataset from webside
             "cl_kaggle_data_graph": "/home/students/meier/MA/MA_Thesis/preprocess/MNLI_test_set_kaggle_graph.csv",
             "cl_kaggle_data_joint": "/home/students/meier/MA/MA_Thesis/preprocess/MNLI_test_set_kaggle_joint.csv",
             "cl_kaggle_data_text": "/home/students/meier/MA/multinli_0.9_test_matched_unlabeled_mod_with_tags_corrected.csv",
             "synthetic_data": "/home/students/meier/MA/inference/generation/extracted_synthetic_generation_data_tags.csv",
             "synthetic_data_joint": "/home/students/meier/MA/data/synthetic_data_joint.csv",
             "synthetic_data_bl": "/home/students/meier/MA/data/extracted_synthetic_generation_data_no_tags.csv"}

    model_path = sys.argv[1]
    outputfile = sys.argv[2]
    model_type = sys.argv[3]

    if model_type == "joint":
        tokenizer.add_tokens(['<t>'], special_tokens=True)
        tokenizer.add_tokens(['</t>'], special_tokens=True)
        tokenizer.add_tokens(['<g>'], special_tokens=True)  # included if necessary
        tokenizer.add_tokens(['</g>'], special_tokens=True)
    if model_type == "text":
        tokenizer.add_tokens(['<t>'], special_tokens=True)
        tokenizer.add_tokens(['</t>'], special_tokens=True)
    if model_type == "graph":
        tokenizer.add_tokens(['<g>'], special_tokens=True)  # included if necessary
        tokenizer.add_tokens(['</g>'], special_tokens=True)

    # new_index = [i for i in range(9847, 19643)] # This index is needed for the kaggle data
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    model = BartForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    df = pd.read_csv(paths["synthetic_data_bl"], delimiter=",")

    tokenized_datasets_test = Dataset.from_pandas(df)
    tokenized_datasets_test = tokenized_datasets_test.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="/")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, compute_metrics=compute_metrics,
                      preprocess_logits_for_metrics=preprocess_logits)  # compute_metrics=compute_metrics
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test)  # ["test"])
    print(res)
    print(res.predictions)
    print(res.metrics)

    final_dataframe = pd.DataFrame({"gold_label": res.predictions})

    final_dataframe.to_csv(outputfile, index=True, header=["gold_label"])
