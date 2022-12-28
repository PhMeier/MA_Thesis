import torch
import pandas as pd

from torch import tensor
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers.pipelines import TextClassificationPipeline
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from transformers import pipeline
import matplotlib.pyplot as plt


class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""

    def __init__(self, name: str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device

    def forward_func(self, inputs: tensor, position=0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                                     attention_mask=torch.ones_like(inputs))
        return pred[position]


    def encode(prem, hypo):
        return tokenizer(prem, hypo, truncation=True,
                         padding='max_length')  # , max_length="max_length")


    def visualize(self, inputs: list, attributes: list):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        attr_sum = attributes.sum(-1)

        attr = attr_sum / torch.norm(attr_sum)

        a = pd.Series(attr.numpy()[0],
                      index=self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().numpy()[0]))

        plt.show(a.plot.barh(figsize=(10, 20)))

    def explain(self, text: str, model):
        """
            Main entry method. Passes text through series of transformations and through the model.
            Calls visualization method.
        """
        prediction = self.__pipeline(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len=inputs.shape[1])

        lig = LayerIntegratedGradients(self.forward_func,  model.base_model.shared)  #getattr(self.__pipeline.model, "base_model").embeddings)

        attributes, delta = lig.attribute(inputs=inputs,
                                          baselines=baseline,
                                          target=self.__pipeline.model.config.label2id[prediction[0]['label']],
                                          return_convergence_delta=True)
        self.visualize(inputs, attributes, prediction)

    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False),
                            device=self.__device).unsqueeze(0)

    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """
        return torch.tensor(
            [self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [
                self.__pipeline.tokenizer.sep_token_id], device=self.__device).unsqueeze(0)


device = "cpu"
sample = "I am very excited about this project"
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model_path = "../checkpoint-3036/"
model = BartForSequenceClassification.from_pretrained(model_path)
x = model.named_parameters()
#for name, para in x:
#    print(name, para)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
exp_model = ExplainableTransformerPipeline(model, clf, device)
exp_model.explain(sample, model)