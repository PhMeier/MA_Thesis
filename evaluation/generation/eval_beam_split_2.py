import csv
import sys
import json
import datasets
import evaluate
import transformers
from datasets import load_dataset
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


def add_EOS_token(example):
    example["premise"] = example["premise"] + " [EOS]"
    return example

if __name__ == "__main__":
    header = ["premise", "hypo", "generated_hypo"]    
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")

    model_path = sys.argv[1] #"/workspace/students/meier/MA/generation/new/bart_67_final/checkpoint-2044/"
    outputfile = sys.argv[2]
    model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    tokenizer.add_tokens(["[EOS]"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))   
 
    dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
    dataset_val = dataset_val.filter(lambda example: example["label"] == 0)
    dataset_val = dataset_val.map(add_EOS_token)
    print(len(dataset_val))
    avg_bleu = 0
    avg_meteor = 0
    avg_bert = 0
    avg_rouge = []
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    for i in range(1500, 3000, 500):
        print(i, i + 500)
        chunk = dataset_val.select(range(i, i + 500))
        hypos = chunk["hypothesis"]
        encoder_input_ids = tokenize_premise(chunk)
        outputs = gen_procedure(encoder_input_ids, model)

        print(*tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False), sep="\n")

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions = preds
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
        with open(outputfile, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for prem, hypo, pred in zip(prems, hypos, predictions):
                x = [prem, hypo, pred]
                writer.writerow(x)
        preds = ""
        outputs = ""
        predictions = ""

    chunk = dataset_val.select(range(3000, len(dataset_val)))
    hypos = chunk["hypothesis"]
    encoder_input_ids = tokenize_premise(chunk)
    outputs = gen_procedure(encoder_input_ids, model)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
    prems = chunk["premise"]
    with open(outputfile, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for prem, hypo, pred in zip(prems, hypos, predictions):
            x = [prem, hypo, pred]
            writer.writerow(x)
    
    print(" --- Final Results ---")
    print("Results BLEU: ", avg_bleu/4)
    print("Results Meteor: ", avg_meteor/4)
    print("Results BERT Score: ", avg_bert/4)
    res = {}
    for item in avg_rouge:
        for key, val in item.items():
            print(key, val)
            print(val)
            # val = float(val)
            if key not in res:
                res[key] = val
            else:
                res[key] += val

    rouge = {k:v / 3 for k, v in res.items()}
    print("Results Rouge: ", rouge)

