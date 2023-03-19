from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")