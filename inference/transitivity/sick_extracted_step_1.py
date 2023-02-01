"""
Evaluate Step 1 of the transitivity task.
Usage:
python3 sick_extracted_step_1.py ./amrbart_text_17/checkpoint-1234 amrbart_text_17_step1_predictions.csv text
"""

from transformers import AutoTokenizer, pipeline, Trainer
import datasets
from datasets import Dataset
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


if __name__ == "__main__":
    paths = {"text": "~/MA/MA_Thesis/data/extracted_sick_with_tags.csv",
             "joint": "/home/students/meier/MA/MA_Thesis/preprocess/sick/extracted_sick_joint.csv"}

    model_path = sys.argv[1]
    outputfile = sys.argv[2]
    model_type = sys.argv[3]  # text or joint as string
    if model_type == "joint":
        tokenizer.add_tokens(['<' + "t" + '>'], special_tokens=True)
        tokenizer.add_tokens(['</' + "t" + '>'], special_tokens=True)
        tokenizer.add_tokens(['<' + "g" + '>'], special_tokens=True)
        tokenizer.add_tokens(['</' + "g" + '>'], special_tokens=True)
    else:
        tokenizer.add_tokens(['<' + "t" + '>'], special_tokens=True)
        tokenizer.add_tokens(['</' + "t" + '>'], special_tokens=True)

    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    model = BartForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    df = pd.read_csv(paths[model_type], delimiter=",")
    tokenized_datasets_test = Dataset.from_pandas(df)
    tokenized_datasets_test = tokenized_datasets_test.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="/")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, preprocess_logits_for_metrics=preprocess_logits,
                      compute_metrics=compute_metrics)  # compute_metrics=compute_metrics
    model.eval()
    res = trainer.predict(tokenized_datasets_test)  # ["test"])
    print(res)
    print(res.predictions)
    print(res.metrics)
    final_dataframe = pd.DataFrame({"index": list(range(len(tokenized_datasets_test))), "label": res.predictions})
    final_dataframe.to_csv(outputfile, index=False, header=["index", "label"])

