import pyonmttok
import tensorflow as tf


SPKO_MODEL_PATH = "./export-sp/spkoD.model"
SPEN_MODEL_PATH = "./export-sp/spenD.model"
KOEN_NMT_PATH = "./export-koen"
ENKO_NMT_PATH = "./export-enko"

class OwlTranslation(object):
    def __init__(self):
        print("[LOG] open NMT loading start ... ")
        self._koTokenizer = pyonmttok.Tokenizer("none", sp_model_path=SPKO_MODEL_PATH)
        self._enTokenizer = pyonmttok.Tokenizer("none", sp_model_path=SPEN_MODEL_PATH)
        self._imported = tf.saved_model.load(KOEN_NMT_PATH)
        self._koen_translate_fn = self._imported.signatures["serving_default"]
        self._imported = tf.saved_model.load(ENKO_NMT_PATH)
        self._enko_translate_fn = self._imported.signatures["serving_default"]
        print("LOG - open NMT loading end, ready! ")

    def _preprocess(self, tokenizer, texts):
        all_tokens = []
        lengths = []
        max_length = 0
        for text in texts:
            tokens, _ = tokenizer.tokenize(text)
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

    def _postprocess(self, tokenizer, outputs):
        texts = []
        for tokens, length in zip(outputs["tokens"].numpy(), outputs["length"].numpy()):
            tokens = tokens[0][: length[0]].tolist()
            texts.append(tokenizer.detokenize(tokens))
        return texts


    def translateKoEn(self, texts):
        inputs = self._preprocess(self._koTokenizer, texts)
        outputs = self._koen_translate_fn(**inputs)
        return self._postprocess(self._enTokenizer, outputs)

    def translateEnKo(self, texts):
        inputs = self._preprocess(self._enTokenizer, texts)
        outputs = self._enko_translate_fn(**inputs)
        return self._postprocess(self._koTokenizer, outputs)

'''
#TEST CODE
def main():

    translator = OwlTranslation()

    while True:
        text = input("Source Korean: ")
        output = translator.translateKoEn([text])
        print("Target: %s" % output[0])
        print("")

        text = input("Source English: ")
        output = translator.translateEnKo([text])
        print("Target: %s" % output[0])
        print("")


if __name__ == "__main__":
    main()


'''


'''
    def testTFLiteInterpreter(self, model, params=None, quantization=None):
        if params is None:
            params = {}
        export_dir = self.get_temp_dir()
        _convert_tflite(model, export_dir, params, quantization)
        self.assertTrue(dir_has_tflite_file(export_dir))
        export_file = os.path.join(export_dir, "opennmt.tflite")
        
        interpreter = tf.lite.Interpreter(model_path=export_file, num_threads=1)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_ids = [2, 3, 4, 5, 6]
        interpreter.resize_tensor_input(0, [len(input_ids)], strict=True)
        interpreter.allocate_tensors()
        np_in_data = np.array(input_ids, dtype=np.int32)
        interpreter.set_tensor(input_details[0]["index"], np_in_data)
        interpreter.invoke()
        output_ids = interpreter.get_tensor(output_details[0]["index"])
        return output_ids



'''