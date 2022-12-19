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
        num_beams=5,
        num_return_sequences=5,
        no_repeat_ngram_size=1,
        early_stopping=True,
        remove_invalid_values=True,
        repetition_penalty=1.2)
    return outputs



if __name__ == "__main__":
    
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")

    model_path = sys.argv[1] #"/workspace/students/meier/MA/generation/new/bart_67_final/checkpoint-2044/"

    model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
    dataset_val = dataset_val.filter(lambda example: example["label"] == 0)
    print(len(dataset_val))
    avg_bleu = 0
    avg_meteor = 0
    avg_bert = 0
    avg_rouge = []
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    for i in range(0, 3250, 250):
        print(i, i + 250)
        chunk = dataset_val.select(range(i, i + 250))
        hypos = chunk["hypothesis"]
        encoder_input_ids = tokenize_premise(chunk)
        outputs = gen_procedure(encoder_input_ids, model)

        print(*tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False), sep="\n")

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions = preds[::5]
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
        avg_rouge.append([rouge_dict])
        print("\n ---- Results ---- \n")
        print("Results BertScore: \n", results_bert)
        print("Average Bert: ", results_bert_average)
        print("Bleu Scores: ", results_bleu)
        print("Meteor Score: ", results_meteor)
        for key, val in results_rouge.items():
            print(key, val)
        preds = ""
        outputs = ""
        predictions = ""

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

    print(" --- Final Results ---")
    print("Results BLEU: ", avg_bleu/14)
    print("Results Meteor: ", avg_meteor/14)
    print("Results BERT Score: ", avg_bert/14)
    res = {}
    for d in results_rouge:
        res.update(d)
    rouge = {k:v / 14 for k, v in res}
    print("Results Rouge: ", rouge)

