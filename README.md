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

1.1. Web-based System Architecture

<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/048e6b35-b3fe-4396-99b3-fba2f9c5c9a7" width="500"/>

1.2. Machine Learning based Architecture from Scrath

<img src="https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/78391f09-8c89-442b-82ab-16d0fb48d0f3" width="500"/>

1.3. Machine Learning based Architecture from fine-tuning based on LLM

![295A_SystemArchitectureLLM](https://github.com/YoonjungChoi/CMPE295_NMT_Project/assets/20979517/6d08f56e-8651-4187-b726-aed215c00fb3)

1.4. Transformer Architecture

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

## 3. Scratch Model with OpenNMT-tf toolkit

### 3.1 Data Collection

We collected dataset Netflix, Ted, Kaggle, Tatoeba, WMT, KoNLP, or website so on.

This [file](https://github.com/YoonjungChoi/CMPE295_NMT_Project/blob/main/fromScratchModels/data_preprocessing/Collected_Dataset_Information.pdf) indicate all sources information.

**Data Augmentation with existing HuggingFaceSeqToSeqLM for inference, Back Translation** 

Collect Mono lingual dataset

> 1.1 download - WMT En-De dataset having total 4,520,346 lines
```
4,562,102 wmt.en
4,520,346 wmtclean.en  (remove HTTP stuff)
```
> 1.2 download Korean dataset havinf total  5,882,096 lines
```
(base) ➜  FINAL wc -l *.ko
  294049 konlp.test.ko
 5293998 konlp.train.ko
  294049 konlp.valid.ko
```

> 2. inference

[HuggingFace EnKo/KoEn model](https://github.com/QuoQA-NLP/T5_Translation) - inference to make synthetic dataset


The last dataset number of lines
```
    557778 test.tok.en
    557778 test.tok.ko
  10041133 train.tok.en
  10041133 train.tok.ko
    557778 valid.tok.en
    557778 valid.tok.ko
```

A total of 11,156,689 lines were collected and divided into 95% for training and 5% for validation and testing. 


### 3.2 Training, Experiments Commands

Doc link - https://opennmt.net/OpenNMT-tf/

Sentence Piece Scripts [file](https://github.com/YoonjungChoi/CMPE295_NMT_Project/blob/main/fromScratchModels/script/hpc/sentencepiece_script.sh)

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

#tokenization
$ spm_encode --model=spenD.model < raw/train.ko > data/train.tok.ko

#detokenization
$ spm_decode --model=spenD.model --input_format=piece < pred.tok.avg.en > pred.tok.avg.en.detok

#evaluation BLEU score
$ sacrebleu raw_test.en < pred.tok.en.avg.detok
$ sacrebleu raw_test.en < pred.tok.en.avg.detok --metrics chrf
```

### 3.3 Scores

Whenever we collected enough dataset, we trained and evaluated models.

Scores are all recorded at this [file-scores](https://github.com/YoonjungChoi/CMPE295_NMT_Project/blob/main/fromScratchModels/data_preprocessing/scores.pdf)

We start training models with Transformer architecture using OpenNMT Toolkit with parameters; beam width, early stopping conditions, maximum lengths so on. 

The final models are averaged from the last 8 models.

| 557,778 testset  | En To Ko | Ko To En |
|------------------|----------|----------|
| Bleu Score       |   34.4   |   40.2   |


Due to limitations in the quality of the dataset, our scratch models are saturated in the scores 35~40. Also, for the record, this dataset includes unbalanced data between general bitext and idiom bitext.

The existing other models have obtained higher scores up to 45, so we would rather use them as a baseline model. 

## 4. Fine-tuned model based on Huggingface Quoqa-NLP models.

**(Problem/Solution)** Existing translation models cannot translate “idiom” expression. Also, We cannot use other models for back-translation to create additional synthetic datasets in case of idioms. To solve this problem, First, we need to collect idiom datasets. Second, we can fine-tune and optimize pretrained models to be available for idiom expressions. Third, this models can be used for idioms centric data augmentation between Korean and English. We want to focus on “Idiom-Centric Data Augmentation Models”.


### 4.1 Test about Idioms expression on baseline model

```
samples
[ I will play it by ear, I've got butterflies in my stomach,
The crowd went bananas when the concert began,
Danny's family told him to “break a leg” right before he went up on stage,
Things quickly went south when my phone got hacked,
유유상종입니다,
내 코가 석자다,
진퇴양란이다,
쥐구멍에도 볕 들 날 있다,
영철이 완전 개천에서 용난 케이스야,
식은 죽 먹기다 ] 

are translated as

[ 나는 그것을 귀로 연주할 것입니다,
저는 배에 나비가 생겼어요,
콘서트가 시작되자 관중들은 바나나를 먹었다,
대니의 가족은 그가 무대에 오르기 직전에 “다리를 끊어라" 고 했다,
내 핸드폰이 해킹을 당하자 상황이 빠르게 남갔다,
It's Yuyusangjong,
I can't help you because my nose is a stone,
It's a dysphagia,
There is a sun in the mouse hole, let's try hard,
It's a case where Yeongchul is in a full stream,
Food is eating porridge.] 
```

Baseline inference [results](https://github.com/YoonjungChoi/CMPE295_NMT_Project/blob/main/finetunedModels/hugginface/infer_results/baseline_infer_results.csv)

### 4.2 Collect BiText of Idiom expression 

Collected BiText Idioms from seaching and some websites [kaggle_scraping_idioms_english](https://www.kaggle.com/code/sohaelshafey/english-idioms-from-url-to-csv), [800_most_commonly_used_idioms_englsih](https://www.academia.edu/11281938/The_800_Most_Commonly_Used_Idioms_in_America)

saved in [here-dataset](https://github.com/YoonjungChoi/CMPE295_NMT_Project/tree/main/finetunedModels/hugginface/data)

### 4.3 Train with finetuning and Evaluation

The baseline models show their performance on the Idiom test set. The English to Korean model shows a score of 7.81 and the Korean to English model shows a score of 17.45.
From this baseline models, we fine-tuned the models by optimizing the hyperparameters with 10 epochs, 5 beam widths, 512 maximum token lengths, 1.3 repetition penalty, 3 repetition ngram size, Adam as the optimization algorithm, 2 gradient accumulation, and 64 batch size for the train. Base on the Idiom test set, the English to Korean model shows a score of 16.98 and the Korean to English model shows a score of 32.72.

| 120 testset | En To Ko | Ko To En |
|-------------|----------|----------|
| Baseline    |   7.81   |   17.45  |
| Finetuned   |   16.98  |   32.72  |


```
Fine-tuned models inference results:
[ 제가 유동적으로 조정할 것입니다,
가슴이 두근두근합니다,
콘서트가 시작했을 때 군중은 열광했습니다,
대니의 가족은 그가 무대에 오르기 직전 그에게 “대박나라”고 말했습니다,
제 전화기가 해킹당했을 때 상황이 빠르게 악화되었습니다,
Birds of a feather flock together.,
I have my own fish to fry,
It's between the devil and the deep blue sea,
Every dog has his day,
Youngchul is a case of rags to riches,
It's a piece of cake.]
```


## 5 Web Application Demo for Project EXPO

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




