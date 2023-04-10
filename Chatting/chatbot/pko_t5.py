from transformers import T5TokenizerFast, T5ForConditionalGeneration

class Pko_T5:
    def __init__(self, model_path, tokenizer_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
        
    def add_tokens(self, tokens=["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "(여자)화자"]):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def make_answer(self, query):
        input_ids = self.tokenizer(query, max_length=32, truncation=True, return_tensors='pt')
        output_ids = self.model.generate(**input_ids,
                                         num_beams=2,
                                         do_sample=True,
                                         min_length=10,
                                         max_length=32,
                                         no_repeat_ngram_size=2)
        result = str(self.tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_space=True))
        answer = result.replace('[', '').replace(']', '').replace("'", '').replace("'", '')
        return answer