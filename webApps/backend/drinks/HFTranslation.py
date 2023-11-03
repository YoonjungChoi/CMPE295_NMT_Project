from transformers import AutoTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM
from easydict import EasyDict
import yaml

print("[LOG] import done")
ENKO_NMT_PATH = "./results_En2Ko/"
KOEN_NMT_PATH = "./results_Ko2En/"


class MyHFTranslation(object):
    
    def __init__(self):
        print("[LOG] MyHFTranslation init..")
        self._koEnTokenizer = AutoTokenizer.from_pretrained(KOEN_NMT_PATH)
        self._koEnNMT = AutoModelForSeq2SeqLM.from_pretrained(KOEN_NMT_PATH)
        
        self._enKoTokenizer = AutoTokenizer.from_pretrained(ENKO_NMT_PATH)
        self._enKoNMT = AutoModelForSeq2SeqLM.from_pretrained(ENKO_NMT_PATH)
        print("[LOG] NMT modules' initialization is done!")

    
    def _translate(self, src_text, tokenizer, model):
        translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding='max_length', max_length=512),
                                    max_length=512,
                                    num_beams=5,
                                    repetition_penalty=1.3,
                                    no_repeat_ngram_size=3,
                                    num_return_sequences=1,)
        
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    
    def translateKoEn(self, texts):
        return self._translate(texts, self._koEnTokenizer, self._koEnNMT)
    
    
    def translateEnKo(self, texts):
        return self._translate(texts, self._enKoTokenizer, self._enKoNMT)
    

# == == == = = == TEST = = =======

def main():
    translator = MyHFTranslation()
    while True:
        text = input("Source English: ")
        output = translator.translateEnKo(text)
        print("Target: %s" % output[0])
        
        text = input("Source Korean: ")
        output = translator.translateKoEn(text)
        print("Target: %s" % output[0])

if __name__ == "__main__":
    main()


