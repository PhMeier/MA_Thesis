"""
This script is used to analyze thee predictions of BART models using captum
https://captum.ai/tutorials/Bert_SQUAD_Interpret

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BartForSequenceClassification, BertConfig, AutoTokenizer

from captum.attr import visualization as viz, InputXGradient, LayerGradientXActivation
from captum.attr import LayerConductance, LayerIntegratedGradients


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
                    [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids,
                                                 position_ids=ref_position_ids)

    return input_embeddings, ref_input_embeddings



def predict(inputs, position_ids=None, attention_mask=None):
    output = model(inputs, attention_mask=attention_mask)
    return output


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    """
    Custom Forward function to grant explainability
    :param inputs:
    :param token_type_ids:
    :param position_ids:
    :param attention_mask:
    :param position:
    :return:
    """
    pred = predict(inputs,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    #output = pred.logits
    #logits = output[0]
    #logits = logits.argmax(dim=-1)
    #pred = pred[position]
    logits = pred.logits
    return logits #pred.max(1).values

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# Start here

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
model_path = "../checkpoint-3036/"

# load model
model = BartForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequenc

premise, hypothesis = "Nike declined to be a sponsor", "Nike is a sponsor."
premise, hypothesis = "We've got to find Tommy.", "We've found tommy."

input_ids, ref_input_ids, sep_id = construct_input_ref_pair(premise, hypothesis, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

ground_truth = 'Nike is a sponsor.'
true_label = 0

ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1

start_scores = predict(input_ids,position_ids=position_ids,attention_mask=attention_mask)
end_scores=8
y = start_scores[0]
print(y)
output = start_scores.logits
logits = output[0]
logits = logits.argmax(dim=-1)
logits = logits.item()

label_dict = {0:"Entailment", 1:"Neutral", 2:"Contradiction"}

print('Predicted Answer:: ', label_dict[logits])
#print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

# Attributions for embedding layers
x = model.named_parameters()

lig = LayerGradientXActivation(squad_pos_forward_func, model.base_model.shared)

attributions_start = lig.attribute(inputs=input_ids,
                                  additional_forward_args=(token_type_ids, position_ids, attention_mask, 0),
                                  target =true_label)
attributions_end = lig.attribute(inputs=input_ids,
                                additional_forward_args=(token_type_ids, position_ids, attention_mask, 1),
                                target=true_label)


attributions_start_sum = summarize_attributions(attributions_start)
attributions_end_sum = summarize_attributions(attributions_end)



# storing couple samples in an array for visualization purposes
start_position_vis = viz.VisualizationDataRecord(
                        attributions_start_sum, # word attr
                        torch.max(torch.softmax(start_scores[0], dim=0)), # pred prob
                        torch.argmax(start_scores[0]), # pred class
                        true_label, # True class
                        2,#str(ground_truth_start_ind), # attr class
                        attributions_start_sum.sum(), # attr class
                        all_tokens, #
                        0)
"""
end_position_vis = viz.VisualizationDataRecord(
                        attributions_end_sum,
                        torch.max(torch.softmax(end_scores[0], dim=0)),
                        torch.argmax(end_scores),
                        torch.argmax(end_scores),
                        str(ground_truth_end_ind),
                        attributions_end_sum.sum(),
                        all_tokens,
                        delta_end)
"""
print('\033[1m', 'Visualizations For Start Position', '\033[0m')
#viz.visualize_text([start_position_vis])
x = viz.visualize_text([start_position_vis])
#print('\033[1m', 'Visualizations For End Position', '\033[0m')
#viz.visualize_text([end_position_vis])
with open("input_x_gradient_Tommy.html", "w", encoding="utf-8") as f:
    f.write(x.data)


