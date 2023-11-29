#!/bin/bash

#SBATCH --job-name=YJEK
#SBATCH --output=log.infer.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-user=yoonjung.choi@sjsu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --partition=gpu
#SBARCH --exclusive

echo ':: Start ::'
source ~/.bashrc
module load anaconda/3.9
#module load cuda/11.2
#module load cudnn8.1-cuda11.2/8.1.1.33
module load cudnn8.1-cuda11.2/8.1.1.33
#module load cuda12.0
#module load nccl2-cuda11.2-gcc9/2.14.3
conda activate wmt_infer
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python3 infer.py
python3 train.py
echo ':: End ::'



