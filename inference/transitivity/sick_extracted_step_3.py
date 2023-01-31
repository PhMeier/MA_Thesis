from transformers import AutoTokenizer, pipeline, Trainer
import datasets
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import pipeline, TrainingArguments
#import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
CUDA_LAUNCH_BLOCKING=1
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
    paths = {"text_neg": "/home/students/meier/MA/inference/sick/step_3_text_extractions_neg.csv",
             "text_pos": "/home/students/meier/MA/inference/sick/step_3_text_extractions_pos.csv",
             "text_comm": "/home/students/meier/MA/inference/sick/step_3_text_extractions_common.csv",
             "text_neutEntNeut": "/home/students/meier/MA/inference/sick/text_neut_ent_neut.csv",
             "text_neutNeutNeut": "/home/students/meier/MA/inference/sick/text_neut_neut_neut.csv",
             "joint_pos": "/home/students/meier/MA/inference/sick/step_3_joint_extractions_pos.csv",
             "joint_comm": "/home/students/meier/MA/inference/sick/step_3_joint_extractions_common.csv",
             "joint_neg": "/home/students/meier/MA/inference/sick/step_3_joint_extractions_neg.csv",
             "joint_neutEntNeut": "/home/students/meier/MA/inference/sick/joint_neut_ent_neut.csv",
             "joint_neutNeutNeut": "/home/students/meier/MA/inference/sick/joint_neut_neut_neut.csv"}
    
    model_path = sys.argv[1]
    outputfile = sys.argv[2]
    model_type = sys.argv[3] # text or joint as string
    pos_neg = sys.argv[4]    
    if model_type == "joint":
        tokenizer.add_tokens(['<' + "t" + '>'], special_tokens=True)
        tokenizer.add_tokens(['</' + "t" + '>'], special_tokens=True)
        tokenizer.add_tokens(['<' + "g" + '>'], special_tokens=True)
        tokenizer.add_tokens(['</' + "g" + '>'], special_tokens=True)
    else:
        tokenizer.add_tokens(['<' + "t" + '>'], special_tokens=True)
        tokenizer.add_tokens(['</' + "t" + '>'], special_tokens=True)


    #new_index = [i for i in range(9847, 19643)] # This index is needed for the kaggle data
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    # /workspace/students/meier/MA/SOTA_Bart/best
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    df = pd.read_csv(paths[model_type+"_"+pos_neg], delimiter=",")
    #dataset_test_split = load_dataset("csv", data_files={"test": paths["cl_kaggle_data"]})
    #dataset_test_split = load_dataset("glue", "mnli", split='test_matched')
    tokenized_datasets_test = Dataset.from_pandas(df)
    #dataset_test_split = dataset_test_split.remove_columns("label")
    #tokenized_datasets_test = dataset_test_split.rename_column("signature", "label")
    #tokenized_datasets_test = dataset_test_split.rename_column("sentence1", "premise")
    #tokenized_datasets_test = tokenized_datasets_test.rename_column("sentence2", "hypothesis")
    tokenized_datasets_test = tokenized_datasets_test.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, preprocess_logits_for_metrics=preprocess_logits,compute_metrics=compute_metrics) #compute_metrics=compute_metrics
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test) #["test"])
    print(res)
    print(res.predictions)
    print(res.metrics)
    #print(res.label_ids.reshape(107, 14).tolist())
    final_dataframe = pd.DataFrame({"index": list(range(len(tokenized_datasets_test))), "label": res.predictions})
    final_dataframe.to_csv(outputfile, index=False, header=["index", "label"])
    #final_dataframe.DataFrame(res.predictions).to_csv("results_mnli_matched_kaggle_bartLarge.csv")
    #print(res.metrics)