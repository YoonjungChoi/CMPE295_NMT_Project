# CMPE295_NMT_Project
[Document](https://docs.google.com/document/d/1Tm7Ttn58zOEZKsSziA-wT1ZTithQNs6O/edit?usp=sharing&ouid=118008271487839144751&rtpof=true&sd=true)
- Literature Search, State of the Art
- Project Justification
- Identify Baseline Approaches
- Dependencies and Deliverables
- Project Architecture
- Evaluation Methodology
- System Design/Methodology
- Implementation Plan and Progress
- Project Schedule

## Architecture

[all diagrams have done here](https://drive.google.com/file/d/1M_D2lVIAyuQwGH5OIrTGJxaz1VKl94gw/view?usp=sharing)

System Architecture

<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/048e6b35-b3fe-4396-99b3-fba2f9c5c9a7" width="500"/>

Machine Learning Architecture

<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/78391f09-8c89-442b-82ab-16d0fb48d0f3" width="500"/>


## HPC Usages 

SJSU HPC Access Instructions => https://www.sjsu.edu/cmpe/resources/hpc.php

1. request HPC account and access through Professor
2. install VPN Connection
3. connect via terminal
 ```
   ssh SJSU_ID@coe-hpc.sjsu.edu
 ```
4. command list
  ```
    sbatch run.sh
    ssh g4 'nvidia-smi'
    module load python3
  ```
5. script
```
#!/bin/bash

#SBATCH --job-name=**YOURS**
#SBATCH --output=**YOURS**
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-user=**YOURS**
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --partition=gpu

echo ':: Start ::'
source ~/.bashrc
module load anaconda/3.9
module load cuda12.0
'''
write things
''''
echo ':: End ::'
```






