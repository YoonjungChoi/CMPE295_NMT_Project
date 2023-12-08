import argparse
import os
import pyonmttok
import tensorflow as tf

SP_ENKO = "data/spenkoD.model"
SP_KOEN = "data/spkoenD.model"

class Translator(object):
    def __init__(self, src_tok, trg_tok, ):
        self._src_tokenizer = pyonmttok.Tokenizer("none", sp_model_path=src_tok)
        self._trg_tokenizer = pyonmttok.Tokenizer("none", sp_model_path=trg_tok)

    def translate(self, texts):
        """Translates a batch of texts."""
        inputs = self._preprocess(texts)
        outputs = self._translate_fn(**inputs)
        return self._postprocess(outputs)

    def _preprocess(self, texts):
        all_tokens = []
        lengths = []
        max_length = 0
        for text in texts:
            tokens, _ = self._tokenizer.tokenize(text)
            length = len(tokens)
            all_tokens.append(tokens)
            lengths.append(length)
            max_length = max(max_length, length)
        for tokens, length in zip(all_tokens, lengths):
            if length < max_length:
                tokens += [""] * (max_length - length)

        inputs = {
            "tokens": tf.constant(all_tokens, dtype=tf.string),
            "length": tf.constant(lengths, dtype=tf.int32),
        }
        return inputs

    def _postprocess(self, outputs):
        texts = []
        for tokens, length in zip(outputs["tokens"].numpy(), outputs["length"].numpy()):
            tokens = tokens[0][: length[0]].tolist()
            texts.append(self._tokenizer.detokenize(tokens))
        return texts


def main():
    parser = argparse.ArgumentParser(description="Translation client example")
    parser.add_argument("--src_tok", required=True, help="src Tokenizer")
    parser.add_argument("--trg_tok", required=True, help="trg Tokenizer")
    parser.add_argument("--model_name", required=True, help="model name")
    args = parser.parse_args()

    translator = Translator(args.src_tok, args.trg_tok, srgs.model_name)

    while True:
        text = input("Source: ")
        output = translator.translate([text])
        print("Target: %s" % output[0])
        print("")


if __name__ == "__main__":
    main()
