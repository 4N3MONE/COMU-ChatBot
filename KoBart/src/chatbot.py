import torch
from src.KoBART import get_kobart_tokenizer
from src.ComuBART import ComuBART

class Chatbot:
    def __init__(self,
                 model_path='src/model_bin',
                 tokenizer_path='src/emji_tokenizer/model.json'):
        self.model = ComuBART(model_path)
        self.tokenizer = get_kobart_tokenizer(tokenizer_path)
        #self.tokenizer.add_tokens(['₩화자₩', '₩청자₩', '₩(남자)화자₩', '₩(여자)화자₩', '₩(남자)청자₩', '₩(여자)청자₩'])
    
    def chat(self, query):
        inputs =  torch.tensor([[self.tokenizer.bos_token_id] + self.tokenizer.encode(query) + [self.tokenizer.eos_token_id]])
        outputs = self.model.generate(inputs)
        answer = self.tokenizer.batch_decode(outputs.tolist())[0]
        answer = answer.replace('<s>', '').replace('</s>', '')
        return answer