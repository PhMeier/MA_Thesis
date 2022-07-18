"""
The AMR-Bart training does not include the computation of metrics like accuracy, only the loss is computed.
In order to find the best checkpoint, it is necessary to compute the scores on the validation set. This is done here.

@date 18/07/2022
@author Philipp Meier
"""
import datasets
import numpy as np
from datasets import load_dataset
from transformers import Trainer
from transformers import AutoTokenizer, BartForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


if __name__ == "__main__":
    paths = {"cl_model": "/workspace/students/meier/MA/AMR_Bart/best/checkpoint-6136/",
             "tow_model": "../checkpoint-12000/",
             "tow_data": "C:/Users/Meier/Projekte/MA_Thesis/preprocess/verb_verid_nor.csv"}

    # /workspace/students/meier/MA/SOTA_Bart/best
    #path = "../checkpoint-12000/"  # "../checkpoint-12000/"
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(paths["cl_model"], local_files_only=True)
    dataset_test_split = load_dataset("glue", "mnli", split='validation_matched')
    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    model.eval()
    res = trainer.predict(tokenized_datasets_test)
    print(res)