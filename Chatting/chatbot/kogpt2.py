from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

class KoGPT2:
    def __init__(self, model_path, tokenizer_path="skt/kogpt2-base-v2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
    def add_tokens(self, tokens=["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "#(여자)화자#"]):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def make_answering(self, query):
        encoded = self.tokenizer.encode('<usr>' + query + '<unused1>' + '0' + '<sys>')
        output_id = self.model.generate(torch.tensor(encoded).unsqueeze(0),
                                        max_length = 30 + len(encoded),
                                        num_beams=2,
                                        top_p = 0.5,
                                        do_sample=True,
                                        no_repeat_ngram_size=2)
        output_list = output_id[0].tolist()
        print(output_list, len(output_list))
        print(self.tokenizer.decode(output_list, skip_special_tokens=True), end='\t')
        answer = self.tokenizer.decode(output_id[0].tolist()[len(encoded):], skip_special_tokens=True)
        return answer