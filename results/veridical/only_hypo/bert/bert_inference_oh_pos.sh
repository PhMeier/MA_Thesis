#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --output=bert_op_67_218574_only_hypo_pos_eval.txt
#SBATCH --mail-user=meier@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH --nodelist=gpu09
#SBATCH --qos=batch

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

suffix=pos
checkp=218574
outputfile=BERT_67_op_only_hypo_${suffix}_${checkp}.csv

# JOB STEPS
source ~/mth_venv/bin/activate
CUDA_LAUNCH_BLOCKING=1
#python3 ../MA_Thesis/inference/veridicality_inference.py #./MA_Thesis/inference/veridicality_inference_kaggle.py
#python3 ../MA_Thesis/inference/veridicality_inference.py
#python3 ../MA_Thesis/inference/inference_csv.py

python3 ../MA_Thesis/inference/bert_test_only_hypo.py ${suffix} ${outputfile} /workspace/students/meier/MA/bert_bw/bert_op_67/checkpoint-${checkp}

python3 ../MA_Thesis/evaluation/veridicality_evaluation.py /home/students/meier/MA/results/only_hypo/bert/${outputfile} ${suffix} ${outputfile} #> /home/students/meier/MA/results/BERT_verid_pos_6070_full_analysis.txt 

deactivate

