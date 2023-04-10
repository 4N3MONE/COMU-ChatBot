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
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.model.save_pretrained(pretrained_path)
    
if __name__=='__main__':
    ckpt_to_pretrained('res_wellness/kobart_chitchat-wellness_model_chp/epoch=09-val_loss=2.740.ckpt', 'kobart_wellness')