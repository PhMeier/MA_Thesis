import datasets
import evaluate
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

#model_path = "/workspace/students/meier/MA/generation/cl_command_line/checkpoint-38000/"
#model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
dataset_val = dataset_val.select(range(20))
dataset_val = dataset_val.filter(lambda example: example["label"]==0)
hypo = dataset_val["hypothesis"]
print(hypo)

print(dataset_val)

def tokenize_premise(example):
    return tokenizer(example["premise"], return_tensors="pt", padding=True).input_ids

#def tokenize_hypothesis(example):
#    return tokenizer(example["hypothesis"], return_tensors="pt").input_ids

encoder_input_ids = tokenize_premise(dataset_val)
#output_ids = tokenize_hypothesis

"""
outputs = model.generate(
    encoder_input_ids,
    num_beams=5,
    num_return_sequences=5,
    no_repeat_ngram_size=1,
    early_stopping=True,
    remove_invalid_values=True,
    repetition_penalty=1.2)

print(*tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False), sep="\n")
"""





