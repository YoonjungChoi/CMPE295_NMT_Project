import torch
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers import Seq2SeqTrainingArguments

print("LOG torch version", torch.__version__)
print("LOG GPU torch.cuda.is_available()", torch.cuda.is_available())
print("LOG GPU torch.cuda.device_count()", torch.cuda.device_count())
print("LOG GPU torch.cuda.current_device()", torch.cuda.current_device())

#device = "cuda:3" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("LOGYJ device:", device)

print("LOGYJ wmt file loading...")

wmtclean = open("mono2.ko", "r")
list_wmtclean = []

for line in wmtclean:
  line=line.strip()
  list_wmtclean.append(line)

wmtclean.close()
df_wmt = pd.DataFrame(list_wmtclean)

print("LOGYJ wmt files loaded done.")

training_args = Seq2SeqTrainingArguments

model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
print("LOGYJ model_name: ", model_name)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
'''
if torch.cuda.device_count()>1:
  print("LOGYJ cuda GPUs")
  #model #i= torch.nn.DataParallel(model)
  model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).to(device)
'''
model.to(device)

print("LOGYJ from pretrained models done")

start = 0
#start = 896000
batch_size = 64 # P100:batch_size 250 / A100:batch_size 700
length = len(df_wmt) #1174045
print("LOGYJ length: ", length)
cnt =  length//batch_size + 1
df = pd.DataFrame(columns = ["src", "gen"])

csv_start = 0
save_start = csv_start
save_count = 0
end = 0

try :
    for i in tqdm(range(start,cnt)):
        save_count+=1
        if i== cnt-1:
            end = length
        else:
            end=csv_start+batch_size

        src_sentences = list_wmtclean[csv_start:end]

        encoding = tokenizer(
          src_sentences, padding=True, return_tensors="pt", truncation=True
        ).to(device)

        # https://huggingface.co/docs/transformers/internal/generation_utils
        with torch.no_grad():
            #translated = model.module.generate(
            translated = model.generate(
              **encoding,
              max_length=64,
              num_beams=5,
              repetition_penalty=1.3,
              no_repeat_ngram_size=3,
              num_return_sequences=1
              #max_length=CFG.max_token_length,
              #num_beams=CFG.num_beams,
              #repetition_penalty=CFG.repetition_penalty,
              #no_repeat_ngram_size=CFG.no_repeat_ngram_size,
              #num_return_sequences=CFG.num_return_sequences,
          )
            del encoding

          # https://github.com/huggingface/transformers/issues/10704
            generated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
            del translated
            print(generated_texts[0:2])

        df1 = pd.DataFrame({"src": src_sentences, "gen": generated_texts})
        #df = df.append(df1, ignore_index = True)
        df = pd.concat([df, df1])
        if save_count == 7000:
            save_count=0
            print("save files...", end)
            df.to_csv(f"./tmp_translated-{save_start}-{end}-sentences.csv", index=False)
        csv_start = end

    if end == length :
        print("end == length")
        df.to_csv(f"./tmp_translated-{save_start}-{end}-sentences.csv", index=False)        

except Exception as e:
    print("Exception", e)
    df.to_csv(f"./tmp_translated-exception-{save_start}-{end}-sentences.csv", index=False)

