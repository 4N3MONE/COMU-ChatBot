import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartConfig, PreTrainedTokenizerFast
import torch

class BartModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart') 
      
    def forward(self):
        pass

def ckpt_to_pretrained(ckpt_path, pretrained_path):
    model = BartModel()
    tokenizer = PreTrainedTokenizerFast(
            tokenizer_file='emji_tokenizer/model.json',
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    tokenizer.add_tokens(["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "#(여자)화자#"])
    model.model.resize_token_embeddings(len(tokenizer))
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.model.save_pretrained(pretrained_path)
    
if __name__=='__main__':
    ckpt_to_pretrained('noes_comuchat/epoch=02-val_loss=4.666.ckpt', '../../Chatting/bin/noes_b_kobart_comuchat')
    ckpt_to_pretrained('noes_comuchat_fm/epoch=02-val_loss=4.704.ckpt', '../../Chatting/bin/noes_b_kobart_comuchat_fm')
    ckpt_to_pretrained('noes_comuchat_ins/epoch=01-val_loss=4.598.ckpt', '../../Chatting/bin/noes_b_kobart_comuchat_ins')
    ckpt_to_pretrained('noes_wellness/epoch=03-val_loss=2.735.ckpt', '../../Chatting/bin/noes_b_kobart_wellness')