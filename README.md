# Predicting Veridicality in a Joint Textual - Symbolic Neural Inference and Generation Architecture

This repository contains the code for the master thesis 'Predicting Veridicality in a Joint Textual-Symbolic Neural 
Inference and Generation Architecture'.

All finetuned models can be found under:
- For MNLI classification: /workspace/students/meier/MA/mnli/
- For veridicality classification and transitivity classification: /workspace/students/meier/MA/Bart_verid/
- For generation tasks: /workspace/students/meier/MA/generation/final_models/

Data on cluster can be found under
- MNLI Classification: /home/students/meier/MA/data/MNLI and /home/students/meier/MA/data/mnli_amr
- Veridicality Classification: /home/students/meier/MA/data/veridicality
- Transitivity Classification: /home/students/meier/MA/data/transitivity
- Generation: /home/students/meier/MA/data/generation


The repository consists of the following subdirectiories:
- evaluation: Scripts for the evaluation of classification and generation tasks. Inference for generation tasks is included in the evaluation.
- fine-tune: Scripts for finetuning Bart-large based models on classification or generation for each task.
- inference: Contains the scripts for the inference for normal NLI, veridicality classification and transitivity
- Paper: Papers used in this thesis.
- preprocess: Preprocessing scripts for creating dataframes to prepare text data for parsing.
- results: Folder contains all the results for classification and generation tasks.
- reports: Contains logs of the fine-tuning procedure.
- utils: Some utility scripts for looking into data.

Example calls are described in the respective file.

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


