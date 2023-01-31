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

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
import sys


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


def construct_whole_bart_embeddings(input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = model.base_model.shared(input_ids)
    ref_input_embeddings = model.base_model.shared(ref_input_ids)

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

def squad_pos_forward_func2(input_emb, attention_mask=None, position=0):
    """
    Used for computation over all layers.
    :param input_emb:
    :param attention_mask:
    :param position:
    :return:
    """
    pred = model(decoder_inputs_embeds=input_emb, attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# Start here

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
model_path = "../checkpoint-3036/"#sys.argv[1] # "../checkpoint-3036/"
outputfile = "bla.html" #sys.argv[2]njklhjk

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

tokenizer.add_tokens(['<g>'], special_tokens=True)  ##This line is updated
tokenizer.add_tokens(['</g>'], special_tokens=True)
tokenizer.add_tokens(['<t>'], special_tokens=True)  ##This line is updated
tokenizer.add_tokens(['</t>'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

#premise, hypothesis = "Nike declined to be a sponsor", "Nike is a sponsor."

#premise, hypothesis = "<g> ( <pointer:0> decline-02 :ARG0 ( <pointer:1> company :wiki <lit> Nike, Inc. </lit> :name ( <pointer:2> name :op1 <lit> Nike </lit> ) ) :ARG1 ( <pointer:3> sponsor-01 :ARG0 <pointer:1> ) )  "," ( <pointer:0> sponsor-01 :ARG0 ( <pointer:1> company :wiki <lit> Nike, Inc. </lit> :name ( <pointer:2> name :op1 <lit> Nike </lit> ) ) ) </g>"

#premise, hypothesis = "<g> ( <pointer:0> attempt-01 :ARG0 ( <pointer:1> country :mod ( <pointer:2> other ) ) :ARG1 ( <pointer:3> get-01 :ARG0 <pointer:1> :ARG1 ( <pointer:4> mandate :purpose ( <pointer:5> administer-01 :ARG0 <pointer:1> :ARG1 ( <pointer:6> country :wiki <lit> State of Palestine </lit> :name ( <pointer:7> name :op1 <lit> Palestine </lit> ) ) ) ) ) ) " , "( <pointer:0> get-01 :ARG0 ( <pointer:1> country :mod ( <pointer:2> other ) ) :ARG1 ( <pointer:3> mandate-01 :ARG1 ( <pointer:4> administer-01 :ARG0 <pointer:1> :ARG1 ( <pointer:5> country :wiki <lit> State of Palestine </lit> :name ( <pointer:6> name :op1 <lit> Palestine </lit> ) ) ) ) ) </g>"

#premise, hypothesis = "<t> We've got to find Tommy. </t> <g> ( <pointer:0> obligate-01 :ARG2 ( <pointer:1> find-01 :ARG0 ( <pointer:2> we ) :ARG1 ( <pointer:3> person :wiki - :name ( <pointer:4> name :op1 <lit> Tommy </lit> ) ) ) ) " ," ( <pointer:0> find-01 :ARG0 ( <pointer:1> we ) :ARG1 ( <pointer:2> person :wiki - :name ( <pointer:3> name :op1 <lit> Tommy </lit> ) ) ) </g>  <t> We've found tommy. </t>"


#premise, hypothesis = "<g> ( <pointer:0> obligate-01 :ARG2 ( <pointer:1> find-01 :ARG0 ( <pointer:2> we ) :ARG1 ( <pointer:3> person :wiki - :name ( <pointer:4> name :op1 <lit> Tommy </lit> ) ) ) ) " ," ( <pointer:0> find-01 :ARG0 ( <pointer:1> we ) :ARG1 ( <pointer:2> person :wiki - :name ( <pointer:3> name :op1 <lit> Tommy </lit> ) ) ) </g>"

#ground_truth = "( <pointer:0> get-01 :ARG0 ( <pointer:1> country :mod ( <pointer:2> other ) ) :ARG1 ( <pointer:3> mandate-01 :ARG1 ( <pointer:4> administer-01 :ARG0 <pointer:1> :ARG1 ( <pointer:5> country :wiki <lit> State of Palestine </lit> :name ( <pointer:6> name :op1 <lit> Palestine </lit> ) ) ) ) ) </g>"  #" ( <pointer:0> find-01 :ARG0 ( <pointer:1> we ) :ARG1 ( <pointer:2> person :wiki - :name ( <pointer:3> name :op1 <lit> Tommy </lit> ) ) )" # "Nike is a sponsor." #"We've found tommy." # 'Nike is a sponsor.'i


#ground_truth = " ( <pointer:0> find-01 :ARG0 ( <pointer:1> we ) :ARG1 ( <pointer:2> person :wiki - :name ( <pointer:3> name :op1 <lit> Tommy </lit> ) ) ) </g>  <t> We've found tommy. </t>"
ground_truth = "Nike is a sponsor."
true_label = 1 #0
attr_label = 2


#premise, hypothesis = "<t> A piece warns that Russia's nuclear power plants are shabby and decaying. </t> <g> ( <pointer:0> warn-01 :ARG0 ( <pointer:1> piece ) :ARG1 ( <pointer:2> and :op1 ( <pointer:3> shabby :domain ( <pointer:4> plant :mod ( <pointer:5> power :mod ( <pointer:6> nucleus ) ) :poss ( <pointer:7> country :wiki <lit> Russia </lit> :name ( <pointer:8> name :op1 <lit> Russia </lit> ) ) ) ) :op2 ( <pointer:9> decay-01 :ARG0 <pointer:4> ) ) )  "," ( <pointer:0> and :op1 ( <pointer:1> shabby ) :op2 ( <pointer:2> decay-01 :ARG0 ( <pointer:3> plant :ARG0-of ( <pointer:4> power-01 :mod ( <pointer:5> nucleus ) ) :poss ( <pointer:6> country :wiki <lit> Russia </lit> :name ( <pointer:7> name :op1 <lit> Russia </lit> ) ) ) ) ) </g>  <t> Russia's nuclear power plants are shabby and decaying. </t>"


#ground_truth = " ( <pointer:0> and :op1 ( <pointer:1> shabby ) :op2 ( <pointer:2> decay-01 :ARG0 ( <pointer:3> plant :ARG0-of ( <pointer:4> power-01 :mod ( <pointer:5> nucleus ) ) :poss ( <pointer:6> country :wiki <lit> Russia </lit> :name ( <pointer:7> name :op1 <lit> Russia </lit> ) ) ) ) ) </g>  <t> Russia's nuclear power plants are shabby and decaying. </t>"

#premise, hypothesis = "<t> And, um, I just wish that my counselor would have been more helpful to me. </t> <g> ( <pointer:0> and :op2 ( <pointer:1> wish-01 :ARG0 ( <pointer:2> i ) :ARG1 ( <pointer:3> helpful-04 :ARG0 ( <pointer:4> person :ARG0-of ( <pointer:5> have-rel-role-91 :ARG1 <pointer:2> :ARG2 ( <pointer:6> counselor ) ) ) :ARG2 <pointer:2> :ARG2-of ( <pointer:7> have-degree-91 :ARG1 <pointer:4> :ARG3 ( <pointer:8> more ) ) ) :mod ( <pointer:9> just ) ) )  ", "( <pointer:0> have-degree-91 :ARG1 <pointer:2> :ARG2 ( <pointer:1> helpful-04 :ARG0 ( <pointer:2> person :ARG0-of ( <pointer:3> have-rel-role-91 :ARG1 ( <pointer:4> i ) :ARG2 ( <pointer:5> counselor ) ) ) :ARG2 <pointer:4> ) :ARG3 ( <pointer:6> more ) ) </g>  <t> My counselor would have been more helpful to me. </t>"

#ground_truth = "( <pointer:0> have-degree-91 :ARG1 <pointer:2> :ARG2 ( <pointer:1> helpful-04 :ARG0 ( <pointer:2> person :ARG0-of ( <pointer:3> have-rel-role-91 :ARG1 ( <pointer:4> i ) :ARG2 ( <pointer:5> counselor ) ) ) :ARG2 <pointer:4> ) :ARG3 ( <pointer:6> more ) ) </g>  <t> My counselor would have been more helpful to me. </t>"

#premise, hypothesis = "<t> I am worried that I should have gotten chocolate ice cream instead of vanilla. </t> <g> ( <pointer:0> worry-02 :ARG0 ( <pointer:1> i ) :ARG1 ( <pointer:2> recommend-01 :ARG1 ( <pointer:3> get-01 :ARG0 <pointer:1> :ARG1 ( <pointer:4> ice-cream :mod ( <pointer:5> chocolate ) :ARG1-of ( <pointer:6> instead-of-91 :ARG2 ( <pointer:7> ice-cream :mod ( <pointer:8> vanilla ) ) ) ) ) :ARG2 <pointer:1> ) )  "," ( <pointer:0> recommend-01 :ARG1 ( <pointer:1> get-01 :ARG0 ( <pointer:2> i ) :ARG1 ( <pointer:3> ice-cream :mod ( <pointer:4> chocolate ) :ARG1-of ( <pointer:5> instead-of-91 :ARG2 ( <pointer:6> ice-cream :mod ( <pointer:7> vanilla ) ) ) ) ) ) </g>  <t> I should have gotten chocolate ice cream instead of vanilla. </t>"

#ground_truth = " ( <pointer:0> recommend-01 :ARG1 ( <pointer:1> get-01 :ARG0 ( <pointer:2> i ) :ARG1 ( <pointer:3> ice-cream :mod ( <pointer:4> chocolate ) :ARG1-of ( <pointer:5> instead-of-91 :ARG2 ( <pointer:6> ice-cream :mod ( <pointer:7> vanilla ) ) ) ) ) ) </g>  <t> I should have gotten chocolate ice cream instead of vanilla. </t>"

#premise, hypothesis = "<t> Sulloway does not contend that firstborn children are nearly always moralistic and far more eager to please their parents than their siblings.  </t> <g> ( <pointer:0> contend-01 :polarity - :ARG0 ( <pointer:1> person :wiki <lit> Gar Sulloway </lit> :name ( <pointer:2> name :op1 <lit> Sulloway </lit> ) ) :ARG1 ( <pointer:3> and :op1 ( <pointer:4> moralistic :domain ( <pointer:5> child :ARG1-of ( <pointer:6> bear-02 :ord ( <pointer:7> ordinal-entity :value 1 ) ) ) :time ( <pointer:8> always :mod ( <pointer:9> near ) ) ) :op2 ( <pointer:10> eager-01 :ARG0 <pointer:5> :ARG1 ( <pointer:11> please-01 :ARG0 <pointer:5> :ARG1 ( <pointer:12> person :ARG0-of ( <pointer:13> have-rel-role-91 :ARG1 <pointer:5> :ARG2 ( <pointer:14> parent ) ) ) ) :ARG2-of ( <pointer:15> have-degree-91 :ARG1 <pointer:5> :ARG3 ( <pointer:16> more :quant ( <pointer:17> far ) ) :ARG4 ( <pointer:18> sibling :poss <pointer:5> ) ) ) ) )  "," ( <pointer:0> and :op1 ( <pointer:1> moralistic :domain ( <pointer:2> child :ARG1-of ( <pointer:3> bear-02 :ord ( <pointer:4> ordinal-entity :value 1 ) ) ) :time ( <pointer:5> always :mod ( <pointer:6> near ) ) ) :op2 ( <pointer:7> eager-01 :ARG0 <pointer:2> :ARG1 ( <pointer:8> please-01 :ARG0 <pointer:2> :ARG1 ( <pointer:9> person :ARG0-of ( <pointer:10> have-rel-role-91 :ARG1 <pointer:2> :ARG2 ( <pointer:11> parent ) ) ) ) :ARG2-of ( <pointer:12> have-degree-91 :ARG1 <pointer:2> :ARG3 ( <pointer:13> more :quant ( <pointer:14> far ) ) :ARG4 ( <pointer:15> sibling :poss <pointer:2> ) ) ) ) </g>  <t> Firstborn children are nearly always moralistic and far more eager to please their parents than their siblings. </t>"

#ground_truth = " ( <pointer:0> and :op1 ( <pointer:1> moralistic :domain ( <pointer:2> child :ARG1-of ( <pointer:3> bear-02 :ord ( <pointer:4> ordinal-entity :value 1 ) ) ) :time ( <pointer:5> always :mod ( <pointer:6> near ) ) ) :op2 ( <pointer:7> eager-01 :ARG0 <pointer:2> :ARG1 ( <pointer:8> please-01 :ARG0 <pointer:2> :ARG1 ( <pointer:9> person :ARG0-of ( <pointer:10> have-rel-role-91 :ARG1 <pointer:2> :ARG2 ( <pointer:11> parent ) ) ) ) :ARG2-of ( <pointer:12> have-degree-91 :ARG1 <pointer:2> :ARG3 ( <pointer:13> more :quant ( <pointer:14> far ) ) :ARG4 ( <pointer:15> sibling :poss <pointer:2> ) ) ) ) </g>  <t> Firstborn children are nearly always moralistic and far more eager to please their parents than their siblings. </t>"

#premise, hypothesis = "<t> Rips does not contend that the rebuttal paper misrepresents the original experiment\'s methods and that it ignores subsequent tests that he regards as immune from data-tuning charges. </t> <g> ( <pointer:0> contend-01 :polarity - :ARG0 ( <pointer:1> person :wiki - :name ( <pointer:2> name :op1 <lit> Rips </lit> ) ) :ARG1 ( <pointer:3> and :op1 ( <pointer:4> misrepresent-01 :ARG0 ( <pointer:5> paper :ARG0-of ( <pointer:6> rebut-01 ) ) :ARG1 ( <pointer:7> method :poss ( <pointer:8> experiment-01 :mod ( <pointer:9> original ) ) ) ) :op2 ( <pointer:10> ignore-01 :ARG0 <pointer:5> :ARG1 ( <pointer:11> test-01 :time ( <pointer:12> subsequent ) :ARG1-of ( <pointer:13> regard-01 :ARG0 <pointer:1> :ARG2 ( <pointer:14> immune-02 :ARG1 <pointer:11> :ARG2 ( <pointer:15> charge-05 :ARG2 ( <pointer:16> tune-01 :ARG1 ( <pointer:17> data ) ) ) ) ) ) ) ) )  "," ( <pointer:0> misrepresent-01 :ARG0 ( <pointer:1> paper :ARG0-of ( <pointer:2> rebut-01 ) ) :ARG1 ( <pointer:3> method :poss ( <pointer:4> experiment-01 :mod ( <pointer:5> original ) ) ) ) </g>  <t> The rebuttal paper misrepresents the original experiment\'s methods. </t>"

#ground_truth = "( <pointer:0> misrepresent-01 :ARG0 ( <pointer:1> paper :ARG0-of ( <pointer:2> rebut-01 ) ) :ARG1 ( <pointer:3> method :poss ( <pointer:4> experiment-01 :mod ( <pointer:5> original ) ) ) ) </g>  <t> The rebuttal paper misrepresents the original experiment\'s methods. </t>"

#premise, hypothesis = "<t> More than one person did not confirm that these statements were correct. </t> <g> ( <pointer:0> confirm-01 :polarity - :ARG0 ( <pointer:1> person :quant ( <pointer:2> more-than :op1 1 ) ) :ARG1 ( <pointer:3> correct-02 :ARG1 ( <pointer:4> thing :ARG1-of ( <pointer:5> state-01 ) :mod ( <pointer:6> this ) ) ) )  "," ( <pointer:0> correct-02 :ARG1 ( <pointer:1> thing :ARG1-of ( <pointer:2> state-01 ) :mod ( <pointer:3> this ) ) ) </g>  <t> These statements were correct. </t>"

#ground_truth = " ( <pointer:0> correct-02 :ARG1 ( <pointer:1> thing :ARG1-of ( <pointer:2> state-01 ) :mod ( <pointer:3> this ) ) ) </g>  <t> These statements were correct. </t>"

#premise, hypothesis = "<t> With a smug smile, Natalia did not confirm that she was already taking care of the issue. </t> <g> ( <pointer:0> confirm-01 :polarity - :ARG0 ( <pointer:1> person :wiki - :name ( <pointer:2> name :op1 <lit> Natalia </lit> ) ) :ARG1 ( <pointer:3> care-03 :ARG0 <pointer:1> :ARG1 ( <pointer:4> issue-02 ) :time ( <pointer:5> already ) ) :manner ( <pointer:6> smile-01 :ARG0 <pointer:1> :manner ( <pointer:7> smug ) ) )  "," ( <pointer:0> care-03 :ARG0 ( <pointer:1> she ) :ARG1 ( <pointer:2> issue-02 ) :time ( <pointer:3> already ) ) </g>  <t> She was already taking care of the issue. </t>"

#ground_truth = " ( <pointer:0> care-03 :ARG0 ( <pointer:1> she ) :ARG1 ( <pointer:2> issue-02 ) :time ( <pointer:3> already ) ) </g>  <t> She was already taking care of the issue. </t>"

#premise, hypothesis = "<t> It was not confirmed that other planes had landed. </t> <g> ( <pointer:0> confirm-01 :polarity - :ARG1 ( <pointer:1> land-01 :ARG1 ( <pointer:2> plane :mod ( <pointer:3> other ) ) ) )  "," ( <pointer:0> land-01 :ARG1 ( <pointer:1> plane :mod ( <pointer:2> other ) ) ) </g>  <t> Other planes had landed. </t>"

#ground_truth = " ( <pointer:0> land-01 :ARG1 ( <pointer:1> plane :mod ( <pointer:2> other ) ) ) </g>  <t> Other planes had landed. </t>"

#premise, hypothesis = "<t> Tax forms have not started to arrive for 2001. </t> <g> ( <pointer:0> start-01 :polarity - :ARG1 ( <pointer:1> arrive-01 :ARG1 ( <pointer:2> form :mod ( <pointer:3> tax-01 ) ) ) :time ( <pointer:4> date-entity :year 2001 ) )  "," ( <pointer:0> arrive-01 :ARG1 ( <pointer:1> form :mod ( <pointer:2> tax-01 ) ) :time ( <pointer:3> date-entity :year 2001 ) ) </g>  <t> Tax forms have arrived for 2001. </t>"

#ground_truth = " ( <pointer:0> arrive-01 :ARG1 ( <pointer:1> form :mod ( <pointer:2> tax-01 ) ) :time ( <pointer:3> date-entity :year 2001 ) ) </g>  <t> Tax forms have arrived for 2001. </t>"

#premise, hypothesis = "<t> The study did not confirm that a standard intervention was suitable for everyone. </t> <g> ( <pointer:0> confirm-01 :polarity - :ARG0 ( <pointer:1> study-01 ) :ARG1 ( <pointer:2> suitable-04 :ARG1 ( <pointer:3> intervene-01 :ARG1-of ( <pointer:4> standard-02 ) ) :ARG2 ( <pointer:5> everyone ) ) )  "," ( <pointer:0> suitable-04 :ARG1 ( <pointer:1> intervene-01 :ARG1-of ( <pointer:2> standard-02 ) ) :ARG2 ( <pointer:3> everyone ) ) </g>  <t> A standard intervention was suitable for everyone. </t>"

#ground_truth = " ( <pointer:0> suitable-04 :ARG1 ( <pointer:1> intervene-01 :ARG1-of ( <pointer:2> standard-02 ) ) :ARG2 ( <pointer:3> everyone ) ) </g>  <t> A standard intervention was suitable for everyone. </t>"

#premise, hypothesis = "<t> He did not have to try something. </t> <g> ( <pointer:0> obligate-01 :polarity - :ARG1 ( <pointer:1> he ) :ARG2 ( <pointer:2> try-01 :ARG0 <pointer:1> :ARG1 ( <pointer:3> something ) ) )  "," ( <pointer:0> try-01 :ARG0 ( <pointer:1> he ) :ARG1 ( <pointer:2> something ) ) </g>  <t> He tried something. </t>"
#ground_truth = " ( <pointer:0> try-01 :ARG0 ( <pointer:1> he ) :ARG1 ( <pointer:2> something ) ) </g>  <t> He tried something. </t>"

#premise, hypothesis = "<t> They argue that scientists are merely desecrating the dead. </t> <g> ( <pointer:0> argue-01 :ARG0 ( <pointer:1> they ) :ARG1 ( <pointer:2> desecrate-01 :ARG0 ( <pointer:3> scientist ) :ARG1 ( <pointer:4> person :ARG1-of ( <pointer:5> die-01 ) ) :mod ( <pointer:6> mere ) ) )  "," ( <pointer:0> desecrate-01 :ARG0 ( <pointer:1> scientist ) :ARG1 ( <pointer:2> person :ARG1-of ( <pointer:3> die-01 ) ) :mod ( <pointer:4> mere ) ) </g>  <t> Scientists are merely desecrating the dead. </t>"

#ground_truth = " ( <pointer:0> desecrate-01 :ARG0 ( <pointer:1> scientist ) :ARG1 ( <pointer:2> person :ARG1-of ( <pointer:3> die-01 ) ) :mod ( <pointer:4> mere ) ) </g>  <t> Scientists are merely desecrating the dead. </t>"

#premise, hypothesis = "<t> Wojtyla argued that intellect is superior to the heart in men. </t> <g> ( <pointer:0> argue-01 :ARG0 ( <pointer:1> person :wiki <lit> Jerzy Juliusz Wojtyla </lit> :name ( <pointer:2> name :op1 <lit> Wojtyla </lit> ) ) :ARG1 ( <pointer:3> superior-01 :ARG1 ( <pointer:4> intellect ) :ARG2 ( <pointer:5> heart :part-of ( <pointer:6> man ) ) ) )  "," ( <pointer:0> superior-01 :ARG1 ( <pointer:1> intellect ) :ARG2 ( <pointer:2> heart :part-of ( <pointer:3> man ) ) ) </g>  <t> Intellect is superior to the heart in men. </t>"

#ground_truth = " ( <pointer:0> superior-01 :ARG1 ( <pointer:1> intellect ) :ARG2 ( <pointer:2> heart :part-of ( <pointer:3> man ) ) ) </g>  <t> Intellect is superior to the heart in men. </t>"

#premise, hypothesis = "<t> Weld also has claimed to be a victim of Helms' ideological extortion. </t> <g> ( <pointer:0> claim-01 :ARG0 ( <pointer:1> person :wiki <lit> William Weld </lit> :name ( <pointer:2> name :op1 <lit> Weld </lit> ) ) :ARG1 ( <pointer:3> victimize-01 :ARG0 ( <pointer:4> extort-01 :ARG0 ( <pointer:5> person :wiki <lit> Jesse Helms </lit> :name ( <pointer:6> name :op1 <lit> Helms </lit> ) ) :mod ( <pointer:7> ideology ) ) :ARG1 <pointer:1> ) :mod ( <pointer:8> also ) )  "," ( <pointer:0> victimize-01 :ARG0 ( <pointer:1> extort-01 :ARG0 ( <pointer:2> person :wiki <lit> Newt Gingrich </lit> :name ( <pointer:3> name :op1 <lit> Gingrich </lit> ) ) :mod ( <pointer:4> ideology ) ) :ARG1 ( <pointer:5> person :wiki <lit> William Weld </lit> :name ( <pointer:6> name :op1 <lit> Weld </lit> ) ) :mod ( <pointer:7> also ) ) </g>  <t> Weld also has been a victim of helms' ideological extortion. </t>"

#ground_truth = " ( <pointer:0> victimize-01 :ARG0 ( <pointer:1> extort-01 :ARG0 ( <pointer:2> person :wiki <lit> Newt Gingrich </lit> :name ( <pointer:3> name :op1 <lit> Gingrich </lit> ) ) :mod ( <pointer:4> ideology ) ) :ARG1 ( <pointer:5> person :wiki <lit> William Weld </lit> :name ( <pointer:6> name :op1 <lit> Weld </lit> ) ) :mod ( <pointer:7> also ) ) </g>  <t> Weld also has been a victim of helms' ideological extortion. </t>"

premise, hypothesis = "<t> Ca\'daan looked out over the desert, hoping to see the young man return. </t> <g> ( <pointer:0> look-01 :ARG0 ( <pointer:1> person :wiki - :name ( <pointer:2> name :op1 <lit> Ca\'daan </lit> ) ) :ARG1 ( <pointer:3> desert ) :manner ( <pointer:4> hope-01 :ARG0 <pointer:1> :ARG1 ( <pointer:5> see-01 :ARG0 <pointer:1> :ARG1 ( <pointer:6> return-01 :ARG1 ( <pointer:7> man :mod ( <pointer:8> young ) ) ) ) ) )  "," ( <pointer:0> look-01 :ARG0 ( <pointer:1> person :wiki - :name ( <pointer:2> name :op1 <lit> Ca\' </lit> :op2 <lit> Daan </lit> ) ) :ARG1 ( <pointer:3> desert ) :direction ( <pointer:4> out ) :time ( <pointer:5> see-01 :ARG0 <pointer:1> :ARG1 ( <pointer:6> return-01 :ARG1 ( <pointer:7> man :mod ( <pointer:8> young ) ) ) ) ) </g>  <t> Ca ` daan looked out over the desert, seeing the young man return. </t>"

ground_truth = " ( <pointer:0> look-01 :ARG0 ( <pointer:1> person :wiki - :name ( <pointer:2> name :op1 <lit> Ca\' </lit> :op2 <lit> Daan </lit> ) ) :ARG1 ( <pointer:3> desert ) :direction ( <pointer:4> out ) :time ( <pointer:5> see-01 :ARG0 <pointer:1> :ARG1 ( <pointer:6> return-01 :ARG1 ( <pointer:7> man :mod ( <pointer:8> young ) ) ) ) ) </g>  <t> Ca ` daan looked out over the desert, seeing the young man return. </t>"

#premise, hypothesis = "<t> She wished that she could return to the past. </t> <g> ( <pointer:0> wish-01 :ARG0 ( <pointer:1> she ) :ARG1 ( <pointer:2> possible-01 :ARG1 ( <pointer:3> return-01 :ARG1 <pointer:1> :ARG4 ( <pointer:4> past ) ) ) )  "," ( <pointer:0> possible-01 :ARG1 ( <pointer:1> return-01 :ARG1 ( <pointer:2> she ) :ARG4 ( <pointer:3> past ) ) ) </g>  <t> She could return to the past. </t>"

#ground_truth = "( <pointer:0> possible-01 :ARG1 ( <pointer:1> return-01 :ARG1 ( <pointer:2> she ) :ARG4 ( <pointer:3> past ) ) ) </g>  <t> She could return to the past. </t>"

#ground_truth = " ( <pointer:0> find-01 :ARG0 ( <pointer:1> we ) :ARG1 ( <pointer:2> person :wiki - :name ( <pointer:3> name :op1 <lit> Tommy </lit> ) ) ) </g>  <t> We've found tommy. </t>"

true_label = 1 #0
attr_label = 1

input_ids, ref_input_ids, sep_id = construct_input_ref_pair(premise, hypothesis, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1

start_scores = predict(input_ids,position_ids=position_ids,attention_mask=attention_mask)
end_scores = 8
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

lig = LayerIntegratedGradients(squad_pos_forward_func, model.base_model.shared)

attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                  baselines=ref_input_ids,
                                  additional_forward_args=(token_type_ids, position_ids, attention_mask, 0),
                                  target =true_label,
                                  return_convergence_delta=True)


attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                additional_forward_args=(token_type_ids, position_ids, attention_mask, 1),
                                target=true_label,
                                return_convergence_delta=True)


attributions_start_sum = summarize_attributions(attributions_start)
attributions_end_sum = summarize_attributions(attributions_end)

print(start_scores[0])
print(torch.max(torch.softmax(start_scores[0], dim=0)))
print(torch.argmax(start_scores[0]))
print(start_scores.logits)
print(torch.argmax(start_scores.logits))

# storing couple samples in an array for visualization purposes
start_position_vis = viz.VisualizationDataRecord(
                        attributions_start_sum, # word attr
                        torch.max(torch.softmax(start_scores[0], dim=0)), # pred prob
                        torch.argmax(start_scores.logits), # pred class
                        true_label,
                        attr_label,# attr class str(ground_truth_start_ind), # true class
                        attributions_start_sum.sum(), # attr class
                        all_tokens, #
                        delta_start)
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

with open(outputfile, "w", encoding="utf-8") as f:
    f.write(x.data)
from IPython.core.display import display, HTML
import imgkit
import matplotlib.pyplot as plt
display(HTML(x.data))




"""
layer_attrs_start = []
layer_attrs_end = []

# The token that we would like to examine separately.
token_to_explain = 23 # the index of the token that we would like to examine more thoroughly
layer_attrs_start_dist = []
layer_attrs_end_dist = []

input_embeddings, ref_input_embeddings = construct_whole_bart_embeddings(input_ids, ref_input_ids, \
                                                                         token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids, \
                                                                         position_ids=position_ids, ref_position_ids=ref_position_ids)

for i in range(model.config.num_hidden_layers):
    lc = LayerConductance(squad_pos_forward_func2, model.base_model.decoder.layers[i])#model.bert.encoder.layer[i])
    layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(attention_mask, 0))
    layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(attention_mask, 1))
    layer_attrs_start.append(summarize_attributions(layer_attributions_start).cpu().detach().tolist())
    layer_attrs_end.append(summarize_attributions(layer_attributions_end).cpu().detach().tolist())

    # storing attributions of the token id that we would like to examine in more detail in token_to_explain
    layer_attrs_start_dist.append(layer_attributions_start[0,token_to_explain,:].cpu().detach().tolist())
    layer_attrs_end_dist.append(layer_attributions_end[0,token_to_explain,:].cpu().detach().tolist())
"""
