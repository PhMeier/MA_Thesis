
import datasets
import numpy as np
from datasets import load_dataset
from transformers import Trainer, BartTokenizer
from transformers import AutoTokenizer, BartForConditionalGeneration
#import jsonlines
import json



tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing")
    mnli_train = load_dataset("glue", "mnli", split='train')
    premises = mnli_train["premise"]
    text = premises[0]
    text = "I am a human."
    res = tokenizer(text, return_tensors="pt", padding="max_length")
    print(res)
    #print(premises[0])

    output = model.generate(input_ids=res["input_ids"], max_length=512)
    print("Output: ", output)
    print(type(output.squeeze()))
    output_text = tokenizer.decode(output.squeeze())
    print(output_text)