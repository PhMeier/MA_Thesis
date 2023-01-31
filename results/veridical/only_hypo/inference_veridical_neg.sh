#!/bin/bash
#SBATCH --job-name=inf_p
#SBATCH --output=amrbart_joint_17_6072_neg_only_hypo_eval_2.txt
#SBATCH --mail-user=meier@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem 8000
#SBATCH --nodelist=gpu09
#SBATCH --qos=bigbatch

suffix=neg
checkp=6072
outputfile=amrbart_17_joint_only_hypo_${suffix}_${checkp}.csv

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/mth_venv/bin/activate
CUDA_LAUNCH_BLOCKING=1
#python3 ../MA_Thesis/inference/veridicality_inference.py #./MA_Thesis/inference/veridicality_inference_kaggle.py
#python3 ../MA_Thesis/inference/veridicality_inference.py

#python3 ../../MA_Thesis/inference/veridicality_inference_only_hypo.py ${suffix} ${outputfile} /workspace/students/meier/MA/Bart_verid/joint/amrbart_mnli_filtered_joint_input_17_rerun/checkpoint-${checkp} cl_joint_${suffix} t

python3 ../../MA_Thesis/evaluation/veridicality_evaluation.py /home/students/meier/MA/results/only_hypo/${outputfile} ${suffix} ${outputfile}_signature_results.txt 
python3 ../../MA_Thesis/evaluation/verid_evaluation_deep.py /home/students/meier/MA/results/only_hypo/${outputfile} ${suffix} #> /home/students/meier/MA/results/BERT_verid_pos_6070_full_analysis.txt


#python3 ../MA_Thesis/inference/veridicality_inference.py ${suffix} ${outputfile} /workspace/students/meier/MA/bert_bw/bert_17/checkpoint-${checkp} cl_data_${suffix}

#python3 ../MA_Thesis/evaluation/veridicality_evaluation.py /home/students/meier/MA/results/${outputfile} ${suffix} ${outputfile}_signature_results.txt 
#python3 ../MA_Thesis/evaluation/verid_evaluation_global.py /home/students/meier/MA/results/${outputfile} ${suffix} #> /home/students/meier/MA/results/BERT_verid_pos_6070_full_analysis.txt


#python3 ../MA_Thesis/inference/inference_csv.py
deactivate
