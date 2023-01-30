import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BartForSequenceClassification, BertConfig, AutoTokenizer

from captum.attr import visualization as viz, TokenReferenceBase
from captum.attr import LayerConductance, LayerIntegratedGradients
import sys


def predict(inputs):
    output = model(inputs)
    return output


def encode(prem, hypo):
    x = tokenizer(prem, hypo, truncation=True)  # , max_length="max_length")
    return torch.tensor([x["input_ids"]], device=device)


vis_data_records_ig = []


def interpret_sentence(model, input_ids, text, min_len=7, label=0):

    model.zero_grad()
    # input_indices dim: [sequence_length]
    seq_length = len(text)

    # predict
    pred = predict(input_ids)
    pred_ind = torch.argmax(pred.logits)
    PAD_IND = tokenizer.pad_token_id
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(inputs=input_ids, baselines=input_ids, target=1,  return_convergence_delta=True)

    print(pred)

    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(viz.VisualizationDataRecord(
        attributions,
        pred,
        torch.argmax(pred.logits), #Label.vocab.itos[pred_ind],
        1, # true
        1,
        attributions.sum(),
        text,
        delta))

model_path = sys.argv[1] # "../checkpoint-3036/"
outputfile = sys.argv[2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
#model_path = "../checkpoint-3036/"#sys.argv[1] # "../checkpoint-3036/"
outputfile = "bla.html" #sys.argv[2]njklhjk

# load model
model = BartForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

premise, hypothesis = "Nike declined to be a sponsor", "Nike is a sponsor."

x = encode(premise, hypothesis)

lig = LayerIntegratedGradients(model,  model.base_model.shared)
text = premise + hypothesis
interpret_sentence(model, x, text)
