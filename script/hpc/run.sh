#!/bin/bash

#SBATCH --job-name=YJ_ENKO
#SBATCH --output=log.train.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --mail-user=yoonjung.choi@sjsu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --partition=gpu

echo ':: Start ::'
source ~/.bashrc
module load anaconda/3.9
module load cudnn8.1-cuda11.2/8.1.1.33
module load cuda12.0
module load nccl2-cuda11.2-gcc9/2.14.3
conda activate env_onmttf
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=0,1,2,3
onmt-main --model_type Transformer --config enko.yaml --auto_config train --with_eval --num_gpus 4
echo ':: End ::'



