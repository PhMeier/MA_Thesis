import datasets
import evaluate
import torch
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, pipeline, Trainer, BartForConditionalGeneration, BartForSequenceClassification, \
    MinLengthLogitsProcessor, LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria, BeamSearchScorer

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

model.config.pad_token_id = model.config.eos_token_id
encoder_input_str ="You and your friends are not welcome here, said Severn." #"Sorry but that's how it is." #"The new rights are nice enough."
hypo = "Severn said the" #people were not welcome there."

min_length = int(len(encoder_input_str.split())*0.3)
print(min_length)
print(len(encoder_input_str.split()))
max_length = (len(encoder_input_str.split()))

input_ids = tokenizer(encoder_input_str, return_tensors="pt")
output = tokenizer(hypo, return_tensors="pt").input_ids #.tolist()

stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
outputs = model.contrastive_search(**input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))