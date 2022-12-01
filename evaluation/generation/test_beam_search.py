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

encoder_input_str ="You and your friends are not welcome here, said Severn." #"Sorry but that's how it is." #"The new rights are nice enough."
hypo = "Severn said the" #people were not welcome there."
encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
output = tokenizer(hypo, return_tensors="pt").input_ids #.tolist()
#"""
min_length = int(len(encoder_input_str.split())*0.3)
print(min_length)
print(len(encoder_input_str.split()))
max_length = (len(encoder_input_str.split()))

"""
outputs = model.generate(
    input_ids,
    max_length=max_length,
    num_beams=2,
    num_return_sequences=2,
    no_repeat_ngram_size=1,
    early_stopping=True,
    remove_invalid_values=True,
    min_length=min_length,
    force_words_ids=output_ids,
    repetition_penalty=1)
"""

"""
# Simple decoding
outputs = model.generate(
    input_ids,
    max_length=max_length,
    min_length=min_length,
    force_words_ids=output_ids,
    do_sample=False,
    top_k=200
)
"""
# Beam Search
num_beams = 3
input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
input_ids = input_ids * model.config.decoder_start_token_id
model_kwargs = {
    "encoder_outputs": model.get_encoder()(
        encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    )
}
beam_scorer = BeamSearchScorer(
    batch_size=1,
    num_beams=num_beams,
    device=model.device,
)
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ]
)
outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
print("Output:\n" + 100 * '-')
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False))