# Importing the T5 modules from huggingface/transformers
from transformers import T5ForConditionalGeneration ,T5TokenizerFast

def init_model(model_name):
    #model_name = 'paust/pko-t5-large'
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model