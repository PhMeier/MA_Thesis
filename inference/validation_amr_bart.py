"""
The AMR-Bart training does not include the computation of metrics like accuracy, only the loss is computed.
In order to find the best checkpoint, it is necessary to compute the scores on the validation set. This is done here.

@date 18/07/2022
@author Philipp Meier
"""
import datasets
import numpy as np
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, BartForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    print(preds)
    print(preds.shape)
    #preds = np.argmax(preds, axis=0)
    result = {}
    print("Preds after argmax: \n ", preds) 
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    print(logits)
    return logits.argmax(dim=1) #-1)


if __name__ == "__main__":
    paths = {"cl_model": "/workspace/students/meier/MA/AMR_Bart/checkpoint-21476/", #best/checkpoint-6136/",
             "tow_model": "../checkpoint-12000/",
             "tow_data": "C:/Users/Meier/Projekte/MA_Thesis/preprocess/verb_verid_nor.csv"}

    # /workspace/students/meier/MA/SOTA_Bart/best
    #path = "../checkpoint-12000/"  # "../checkpoint-12000/"
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(paths["cl_model"], local_files_only=True)
    dataset_test_split = load_dataset("glue", "mnli", split='validation_matched')
    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    model.resize_token_embeddings(len(tokenizer))
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=2, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits, args=targs)
    model.eval()
    res = trainer.predict(tokenized_datasets_test)
    print(res)
