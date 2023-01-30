#!/bin/bash
#SBATCH --job-name=graph_67
#SBATCH --output=generation_bart_mnli_67_graph.txt
#SBATCH --mail-user=meier@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem 32000
#SBATCH --time=2-0:00:00

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/generation_venv/bin/activate
python3 /home/hd/hd_hd/hd_rk435/github_repositories/transformers/examples/pytorch/summarization/run_summarization_amrbart_graph.py \
    --model_name_or_path /pfs/work7/workspace/scratch/hd_rk435-checkpointz/bart_mnli_only_graph_67/checkpoint-2301/ \
    --do_train \
    --do_eval \
    --train_file /pfs/data5/home/hd/hd_hd/hd_rk435/data/generation/MNLI_train_amr_input_generation_hypothesis_is_text.csv \
    --validation_file /pfs/data5/home/hd/hd_hd/hd_rk435/data/generation/MNLI_dev_matched_amr_input_generation_hypothesis_is_text.csv \
    --output_dir /pfs/work7/workspace/scratch/hd_rk435-checkpointz/generation/new/graph_67 \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --max_target_length 1024 \
    --val_max_target_length 1024 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 20 \
    --report_to wandb \
    --fp16 True \
    --save_total_limit 20 \
    --gradient_accumulation_steps 16 \
    --seed 67 \
    --num_beams 5 \
    --tokenizer_name facebook/bart-large \
    --text_column text \
    --summary_column summary \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --predict_with_generate

deactivate
