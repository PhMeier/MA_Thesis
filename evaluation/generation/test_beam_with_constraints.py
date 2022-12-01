import datasets
import evaluate
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, pipeline, Trainer, BartForConditionalGeneration, BartForSequenceClassification, \
    MinLengthLogitsProcessor, LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria, PhrasalConstraint

from transformers import pipeline, TrainingArguments
#import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
from nltk.tokenize import sent_tokenize
CUDA_LAUNCH_BLOCKING=1
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

#model_path = "/workspace/students/meier/MA/generation/SOTA_Bart_Generation_cnn_parameters/checkpoint-3900/"
model_path = "../../data/checkpoint-10200"
model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

encoder_input_str ="You and your friends are not welcome here, said Tom." #"Sorry but that's how it is." #"The new rights are nice enough."
hypo = "Tom said the people were not"# were not welcome there."
input = tokenizer(encoder_input_str, return_tensors="pt").input_ids
output = tokenizer(hypo, return_tensors="pt").input_ids.tolist()
#"""
min_length = int(len(encoder_input_str.split()))
print("Min Len: ", min_length)
print(len(encoder_input_str.split()))
max_length = (len(encoder_input_str.split()))+10
print("Max Length: ", max_length)

constraints = [
    PhrasalConstraint(
        tokenizer(hypo, add_special_tokens=False).input_ids
    )
]

#"""
# Beam search
outputs = model.generate(
    input,
    max_length=max_length,
    num_beams=3,
    do_sample=False,
    num_return_sequences=3,
    no_repeat_ngram_size=1,
    early_stopping=True,
    remove_invalid_values=True,
    min_length=min_length,
    repetition_penalty=1,
    constraints=constraints)
#"""

"""
# Simple decoding
outputs = model.generate(
    input,
    max_length=max_length,
    min_length=min_length,
    force_words_ids=output,
)
"""

"""
# Greedy Decoding
logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(min_length, eos_token_id=model.config.eos_token_id),])
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
outputs = model.greedy_search(input_ids=input, logits_processor=logits_processor, stopping_criteria=stopping_criteria)
"""

print("Output:\n" + 100 * '-')
#print(tokenizer.decode(outputs, skip_special_tokens=True))
print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True))