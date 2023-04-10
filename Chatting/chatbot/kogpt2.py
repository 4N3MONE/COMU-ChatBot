from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

class KoGPT2:
    def __init__(self, model_path, tokenizer_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
    def add_tokens(self, tokens=["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "#(여자)화자#"]):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def make_answering(self, query):
        encoded = self.tokenizer.encode(query)
        input_id = torch.tensor([self.tokenizer.bos_token_id] + encoded + [self.tokenizer.eos_token_id]).unsqueeze(0)
        output_id = self.model.generate(input_id)
        answer = self.tokenizer.decode(output_id[0].tolist()[len(encoded) + 1:], skip_special_tokens=True)
        return answer