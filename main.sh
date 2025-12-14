#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=sft_train
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=demo_1.out

source ~/.bashrc            
conda activate eecs595
export HF_TOKEN="<Hugging_Face_token"

echo "Started on $(hostname) at $(date)"

bash ./scripts/4_run_tot.sh

echo "Job finished at $(date)"