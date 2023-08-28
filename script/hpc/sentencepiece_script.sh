
echo "start"

spm_train --input=raw/train.ko --model_prefix=spkoAB --vocab_size=32000 --character_coverage=1 --model_type=bpe

echo "end spko model, start spen model ... "

spm_train --input=raw/train.en --model_prefix=spenAB --vocab_size=32000 --character_coverage=1 --model_type=bpe

echo "end spen model, start tokenization train.ko "

spm_encode --model=spkoAB.model < raw/train.ko > data/train.tok.ko

echo "end train.ko start valid.ko"
spm_encode --model=spkoAB.model < raw/valid.ko > data/valid.tok.ko
echo "end valid.ko start test.ko"
spm_encode --model=spkoAB.model < raw/test.ko > data/test.tok.ko

echo "end test.ko start train.en"
spm_encode --model=spenAB.model < raw/train.en > data/train.tok.en
echo "end train.en start valid.en"
spm_encode --model=spenAB.model < raw/valid.en > data/valid.tok.en
echo "end valid.en  start test.en"
spm_encode --model=spenAB.model < raw/test.en > data/test.tok.en

echo "end test.en, start formatting"
onmt-build-vocab --from_format sentencepiece --from_vocab spenAB.vocab --save_vocab data/spenAB.onmt.vocab
onmt-build-vocab --from_format sentencepiece --from_vocab spkoAB.vocab --save_vocab data/spkoAB.onmt.vocab

echo "end"
