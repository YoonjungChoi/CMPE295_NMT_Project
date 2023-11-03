# CMPE295_NMT_Project: Idiom-centric data augmentation model on NMT Between Korean and Englsih
[Google Share Document](https://docs.google.com/document/d/1Tm7Ttn58zOEZKsSziA-wT1ZTithQNs6O/edit?usp=sharing&ouid=118008271487839144751&rtpof=true&sd=true)
- Literature Search, State of the Art
- Project Justification
- Identify Baseline Approaches
- Dependencies and Deliverables
- Project Architecture
- Evaluation Methodology
- System Design/Methodology
- Implementation Plan and Progress
- Project Schedule

## 1. Architecture

**REFER PDF BOOK “Speech and Language Processing”**
Available: at [this link](https://web.stanford.edu/~jurafsky/slp3/ed3book_jan122022.pdf)

[all diagrams have done here](https://drive.google.com/file/d/1M_D2lVIAyuQwGH5OIrTGJxaz1VKl94gw/view?usp=sharing)

System Architecture

<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/048e6b35-b3fe-4396-99b3-fba2f9c5c9a7" width="500"/>

Machine Learning Architecture From Scrath

<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/78391f09-8c89-442b-82ab-16d0fb48d0f3" width="500"/>

Machine Learning Architecture Fine-tuning From LLM

![295A_SystemArchitectureLLM](https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/6d08f56e-8651-4187-b726-aed215c00fb3)

Transformer Architecture

!<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/b7701922-20b4-4259-8fcb-f5258266e1ba" width="500" />


## 2. HPC Usages 

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

## 3. Create our own Model with OpenNMT-tf toolkit
doc link - https://opennmt.net/OpenNMT-tf/

```
#training
$ onmt-main --model_type Transformer --config enko.yaml --auto_config train --with_eval --num_gpus 4

#use GPU
$ export CUDA_VISIBLE_DEVICES=0,1,2,3

#inference
$ onmt-main --config enko.yaml --auto_config --checkpoint_path run/avg/ckpt-300000 infer --features_file ../../data/test.tok.en --predictions_file pred.tok.avg.ko

#serving - tflite, servings
# increase beam width up to 5, and export average model

$ onmt-main --config koen.yaml --auto_config average_checkpoints --output_dir run/baseline/avg --max_count 8
$ onmt-main --config enkoExport.yaml --auto_config export --output_dir ./export-tflite/

```

### 3.1 Data Augmentation with existing HuggingFaceSeqToSeqLM for inference, Back Translation

Collect Mono lingual dataset 

1. download - WMT En-De dataset 
```
4562102 wmt.en
4520346 wmtclean.en  (remove HTTP stuff)
```
2. download Korean dataset
```
(base) ➜  FINAL wc -l *.ko
  294049 konlp.test.ko
 5293998 konlp.train.ko
  294049 konlp.valid.ko
 5,882,096 total
```

3. inference

[HuggingFace EnKo/KoEn model] (https://github.com/QuoQA-NLP/T5_Translation) - inference to make synthetic dataset

4. Test about Idioms

```
list_data = ["I will play it by ear.", "I've got butterflies in my stomach.", "The crowd went bananas when the concert began.", "When pigs fly", "I used to get butterflies in my stomach before the tests.", "Things quickly went south when my phone got hacked."]

['나는 그것을 귀로 연주할 것입니다.',
 '저는 배에 나비가 생겼어요.',
 '콘서트가 시작되자 관중들은 바나나를 먹었다.',
 '돼지가 날아가면 돼지가 날아든다.',
 '나는 시험 전에 배에서 나비를 당하기도 했어요.',
 '내 핸드폰이 해킹을 당하자 상황이 빠르게 남갔다.']
```

## 4. Create "Idiom-centric data augmentation Model" by using pretrained-model.

**(Problem/Solution)** Existing translation models cannot translate “idiom” expression. Also, We cannot use other models for back-translation to create additional synthetic datasets in case of idioms. To solve this problem, First, we need to collect idiom datasets. Second, we can fine-tune and optimize pretrained models to be available for idiom expressions. Third, this models can be used for idioms centric data augmentation between Korean and English. We want to focus on “Idiom-Centric Data Augmentation Models”.


## Web Application Demo for Project EXPO

<img width="500" alt="screen shot" src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/62f15c6b-b5de-4bdd-9fe9-f437b6904de1">

**React Frontend**

```
npx create-react-app cmpe298b-project
cd cmpe298b-project
npm start

npm install axios
npm audit fix --force
npm install @material-ui/core 
npm install @mui/material
```

**Django Backend**

```
python3 -m vent .venv

(base) ➜  drinks . .venv/bin/activate
(.venv) (base) ➜  drinks pip install django
(.venv) (base) ➜  drinks pip install djangorestframework
(.venv) (base) ➜  drinks django-admin startproject drinks .
(.venv) (base) ➜  drinks python manage.py runserver

(base) ➜  drinks . .venv/bin/activate
(.venv) (base) ➜  drinks python manage.py migrate
(.venv) (base) ➜  drinks python manage.py createsuperuser
Username (leave blank to use 'yoonjung'): admin
Email address: admin@email.com
Password: admin

python -m pip install django-cors-headers
#POSTMAN should be ON.

(.venv) (base) ➜  drinks pip install pyonmttok
(.venv) (base) ➜  drinks pip install tensorflow
(.venv) (base) ➜  drinks pip install urllib3==1.26.6

```




