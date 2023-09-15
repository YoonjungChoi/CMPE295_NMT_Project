# Create Translation Models with OpenNMT Toolkit 

## 1. Dataset 

We collected from Netflix and other websites from scratch. You can check sources from 'data_source_info.pdf'

Since each data set has different characteristics, preprocessing must be performed.

Netflix's subtitle has some unsupported or unnecessary information:

```
CLEAN : ERROR TYPE
1. - [XX] (XX) ‎ => it should be removed in sentences.
2. remove whole sentence if it includes ♪ 
3. korean doesnt have punctuations to show "end of sentence", 
but English has puctuations like ! . ? <= three punctuations can be used to concat sentences.
```
optional)
4. some cases has "...", which makes it difficult to concat sentences.
To remove :
```
dataframe[0] = dataframe[0].str.replace("\u2026", '', regex=False)
```
<img width="642" alt="Screen Shot 2023-02-17 at 12 26 54 PM" src="https://user-images.githubusercontent.com/20979517/219786533-b1ba4839-7053-4404-b46e-97ef49a1b22c.png">


TATOEBA data set looks alot, but its quality is not good enough to use.

**Idioms** dataset between English and Korean is not easy. 

We collected them from [Idioms site](https://www.theidioms.com/) , [800 idioms PDF](https://www.academia.edu/11281938/The_800_Most_Commonly_Used_Idioms_in_America), [Kaggle English Idioms](https://www.kaggle.com/code/bryanb/scraping-sayings-and-proverbs/notebook#PART-I:-Scraping-English-sayings)

## 2. OpenNMT Toolkit

We did experiments with DatasetA, DatasetA+B, Dataset A+B+C on Transformer, TransformerRelative, TransformerBig Architectures for EnToKo, KoToEn.

There is 'scores.pdf' file to show Bleu scores.

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

## 3. Data Augmentation with existing HuggingFaceSeqToSeqLM for inference, Back Translation

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

## 4. SetUP HPC Environment

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


