from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

def get_kobart_model(model_path=None):
    if not model_path:
        model_path = 'hyunwoongko/kobart'
    model = BartForConditionalGeneration.from_pretrained(model_path)
    return model

def get_kobart_tokenizer(model_path=None):
    if not model_path:
        model_path = 'hyunwoongko/kobart'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_path,
                                    bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    return tokenizer