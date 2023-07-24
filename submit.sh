#!/bin/bash
#SBATCH --job-name=prompt_compressor       # Job name
#SBATCH --nodes=1                    # Number of tasks (processes) to be started
#SBATCH --time=40:00:00                # Walltime limit in hh:mm:ss
#SBATCH --mem=64G              # Memory per CPU core
#SBATCH --gres=gpu:A6000:2
#SBATCH --output=pc_job-%j.out  # Standard output file (%j is replaced with the job ID)
#SBATCH --error=pc_job-%j.err   # Standard error file (%j is replaced with the job ID)

set -xe
echo $CUDA_VISIBLE_DEVICES

python -m torch.distributed.run --nproc_per_node=2 --master_port=1235 finetune.py \ \
    --base_model 'hf_llama_7B' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'