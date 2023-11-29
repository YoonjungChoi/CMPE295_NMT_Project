import pandas as pd
import numpy as np
from easydict import EasyDict
import yaml
from datasets import load_metric, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

def test(data_file, yaml_file) :
    df_test = pd.read_csv(data_file)
    # Read config.yaml file
    with open(yaml_file) as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG["CFG"])

    src_text = df_test[CFG.src_language].values.tolist()
    
    print("[LOG] CFG " , CFG)

    model_name = CFG.inference_model_name
    result_path = CFG.save_path

    tokenizer = AutoTokenizer.from_pretrained(result_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(result_path)
    print("LOG model_name", model_name)
    model.eval()

    translated = model.generate(
        **tokenizer(src_text, return_tensors="pt", padding='max_length', max_length=CFG.max_token_length,),
        max_length=CFG.max_token_length,
        num_beams=CFG.num_beams,
        repetition_penalty=CFG.repetition_penalty,
        no_repeat_ngram_size=CFG.no_repeat_ngram_size,
        num_return_sequences=CFG.num_return_sequences,
    )
    #print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    print("LOG inference is done.")
 
    output = []
    for t in tqdm(translated):
        output.append(tokenizer.decode(t, skip_special_tokens=True))

    output_name = "pred_"+CFG.src_language+CFG.tgt_language
    df_test[output_name] = output
    df_test.to_csv(data_file.split(".")[0]+output_name+".csv")

    print("LOG results file saved.")
    
    preds = df_test[output_name]
    labels = np.expand_dims(df_test[CFG.tgt_language], axis=1)
    
    metric = load_metric("sacrebleu")
    print("LOG metric done")
    score = metric.compute(predictions=preds, references=labels)
    print(score)

def main():
    #test("data_test.csv", "config_enko.yaml")
    test("data/data_test.csv", "config_koen.yaml")
    
if __name__ == "__main__":
    main()