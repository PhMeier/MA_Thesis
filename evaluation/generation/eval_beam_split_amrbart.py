"""
Inference and Evaluation for generation. Dataset is splittd into chunks, since it is too big to process it in one go.
"""
import csv
import pickle
import sys
import json
import datasets
import evaluate
import pandas as pd
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BartForConditionalGeneration


def tokenize_premise(example):
    return tokenizer(example["premise"], return_tensors="pt", padding=True).input_ids


def gen_procedure(encoder_input, model):
    outputs = model.generate(
        encoder_input,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        early_stopping=True,
        remove_invalid_values=True,
        repetition_penalty=1.2)
    return outputs


if __name__ == "__main__":
    header = ["premise", "hypo", "generated_hypo"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model_type_to_data = {"text": "/home/students/meier/MA/inference/generation/data/text/MNLI_dev_mismatched_text_tags_input_generation_hypothesis_is_text_part_1.csv"}
    model_type = sys.argv[1]
    model_path = sys.argv[2]  # "/workspace/students/meier/MA/generation/new/bart_67_final/checkpoint-2044/"
    outputfile = sys.argv[3]
    model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    if model_type == "text":
        num_added_toks = tokenizer.add_tokens(['<t>'], special_tokens=True)  ##This line is updated
        num_added_toks = tokenizer.add_tokens(['</t>'], special_tokens=True)
    if model_type == "graph" or model_type == "joint":
        num_added_toks = tokenizer.add_tokens(['<g>'], special_tokens=True)  ##This line is updated
        num_added_toks = tokenizer.add_tokens(['</g>'], special_tokens=True)
        num_added_toks = tokenizer.add_tokens(['<t>'], special_tokens=True)  ##This line is updated
        num_added_toks = tokenizer.add_tokens(['</t>'], special_tokens=True)
    num_added_toks = tokenizer.add_tokens(['[EOS]'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    df = pd.read_csv(model_type_to_data[model_type], delimiter=",")
    tokenized_datasets_test = Dataset.from_pandas(df)
    dataset_val = tokenized_datasets_test
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")

    #dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
    #dataset_val = dataset_val.filter(lambda example: example["label"] == 0)
    print(len(dataset_val))
    avg_bleu = 0
    avg_meteor = 0
    avg_bert = 0
    avg_rouge = []
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    for i in range(0, len(dataset_val), 577):
        print(i, i + 577)
        chunk = dataset_val.select(range(i, i + 577))
        hypos = chunk["hypothesis"]
        encoder_input_ids = tokenize_premise(chunk)
        outputs = gen_procedure(encoder_input_ids, model)

        #print(*tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False), sep="\n")

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions = preds #[::5]
        print(predictions)
        results_bleu = bleu.compute(predictions=predictions, references=hypos)
        results_bert = bert_score.compute(predictions=predictions, references=hypos, lang="en")
        results_rouge = rouge.compute(predictions=predictions, references=hypos)
        results_meteor = meteor.compute(predictions=predictions, references=hypos)
        results_bert_average = sum(results_bert["precision"]) / len(results_bert["precision"])
        avg_bleu += results_bleu["bleu"]
        avg_bert += results_bert_average
        avg_meteor += results_meteor["meteor"]
        rouge_dict = dict((rn, round(results_rouge[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
        avg_rouge.append(rouge_dict)
        print("\n ---- Results ---- \n")
        print("Results BertScore: \n", results_bert)
        print("Average Bert: ", results_bert_average)
        print("Bleu Scores: ", results_bleu)
        print("Meteor Score: ", results_meteor)
        for key, val in results_rouge.items():
            print(key, val)
        prems = chunk["premise"]
        with open(outputfile, "a", encoding="utf-8", newline="")as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for prem, hypo, pred in zip(prems, hypos, predictions):
                x = [prem, hypo, pred]
                writer.writerow(x)
        preds = ""
        outputs = ""
    """
    chunk = dataset_val.select(range(3250, len(dataset_val)))
    encoder_input_ids = tokenize_premise(dataset_val)
    outputs = gen_procedure(encoder_input_ids, model)
    results_bleu = bleu.compute(predictions=predictions, references=hypos)
    results_bert = bert_score.compute(predictions=predictions, references=hypos, lang="en")
    results_rouge = rouge.compute(predictions=predictions, references=hypos)
    results_meteor = meteor.compute(predictions=predictions, references=hypos)
    results_bert_average = sum(results_bert["precision"]) / len(results_bert["precision"])
    avg_bleu += results_bleu["bleu"]
    avg_bert += results_bert_average
    avg_meteor += results_meteor["meteor"]
    rouge_dict = dict((rn, round(results_rouge[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
    results_rouge.append([rouge_dict])
    """
    n = 3
    print(" --- Final Results ---")
    print("Results BLEU: ", avg_bleu / n)
    print("Results Meteor: ", avg_meteor / n)
    print("Results BERT Score: ", avg_bert / n)
    print("Rouge: ", results_rouge)
    pickle.dump(avg_rouge, open("bart_17_rouge_part_1.p", "wb"))
    res = {}
    for item in avg_rouge:
        for key, val in item.items():
            print(key, val)
            print(val)
            #val = float(val)
            if key not in res:
                res[key]=val
            else:
                res[key] += val
    x = {k:round(v/3,2) for k,v in res.items()}
    print(res)
    print(x)
    #res = {}
    #for d in avg_rouge:
        #res.update(d)
    #rouge = {k: v / n for k, v in res}
    #print("Results Rouge: ", rouge)

