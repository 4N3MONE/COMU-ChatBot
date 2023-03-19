from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

def get_kobart_tokenizer(model_path=None):
    if not model_path:
        model_path = 'hyunwoongko/kobart'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    return tokenizer