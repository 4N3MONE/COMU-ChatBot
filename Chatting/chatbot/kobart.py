from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import torch

class KoBART:
    def __init__(self, model_path, tokenizer_path='chatbot/emji_tokenizer/model.json'):
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
        self.model.eval()
        
    def add_tokens(self, tokens=["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "#(여자)화자#"]):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def make_answering(self, query):
        input_id = torch.tensor([[self.tokenizer.bos_token_id] + self.tokenizer.encode(query) + [self.tokenizer.eos_token_id]])
        output_id = self.model.generate(input_id,
                                        max_length=30,
                                        do_sample = True,
                                        no_repeat_ngram_size=2)
        answer = self.tokenizer.batch_decode(output_id.tolist())[0].replace('<s>', '').replace('</s>', '')
        return answer