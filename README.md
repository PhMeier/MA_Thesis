# Predicting Veridicality in a Joint Textual - Symbolic Neural Inference and Generation Architecture

This repository contains the code for the master thesis 'Predicting Veridicality in a Joint Textual-Symbolic Neural 
Inference and Generation Architecture'.

The repository consists of the following subdirectiories:
- data: Contains the datasets used in this thesis, which are: MNLI dataset, the MNLI_filtered 
  dataset and the test set for veridicality used in [How well do NLI models capture verb veridicality](https://aclanthology.org/D19-1228.pdf), 
  the [SICK dataset](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf). All used data can be found under /home/students/meier/MA/data/
- evaluation: Contains scripts for the evaluation of classification and generation tasks
- fine-tune: Contains scripts for finetuning Bart-large based models on classification or generation for each task.
- inference: Contains the scripts for the inference for normal NLI or veridicality classification
- Paper: Papers used in this thesis
- preprocess: Preprocessing scripts for creating dataframes to prepare text data for parsing
- results: Folder contains all the results for classification and generation tasks
- reports: Contains logs of the fine-tuning procedure.
- utils: Some utility scripts for looking into data.

All finetuned models can be found under
- for MNLI classification: /workspace/students/meier/MA/mnli/
- for veridicality classification and transitivity classification: /workspace/students/meier/MA/Bart_verid/
- for generation tasks: /workspace/students/meier/MA/generation/final_models/

## Requirements
Main requirements are:
- Python 3.7 or higher
- Huggingface 1.19 or higher
- Pytorch 1.12.0 or higher
- NLTK
- Pandas
- SciPy 1.8.0
- Wandb 0.12.18
- Numpy 1.22.3
- spring-amr 1.0

For further information about requirements, please consider the provided requirements.txt file

For classification tasks, four NVIDIA V100 GPUS were used, generation tasks can be performed with one NVIDIA V100 GPU.


