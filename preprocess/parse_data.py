import datasets
import numpy as np
from datasets import load_dataset
from transformers import Trainer
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer
#import jsonlines
import json

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")


def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


if __name__ == "__main__":
    model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing")
    print(tokenizer.bos_token)
    tokenizer.amr_bos_token_id = 0
    print(tokenizer.amr_bos_token_id)
    """
    mnli_train = load_dataset("glue", "mnli", split='train')
    prem = mnli_train[0]["premise"]
    print(prem)
    prem = "The boy wants to go"
    inp = tokenizer.encode_plus(prem, max_length=256, padding=False, truncation=True, return_tensors="pt") #, add_special_tokens=True) #.encode(prem, return_tensors="pt", add_special_tokens=True) #, padding='max_length')
    print(inp)
    output = model.generate(input_ids=inp["input_ids"], max_length=512, num_beams=5, do_sample=False, early_stopping=True)
    print(output[0])
    out = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(out)
    """