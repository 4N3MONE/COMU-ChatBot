from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import os

class MeBART(BartForConditionalGeneration):
    def __init__(self,
                 model_path = 'hyunwoongko/kobart',
                 tokenizer_path = 'hyunwoongko/kobart'):
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.mask_token = '<mask>'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
