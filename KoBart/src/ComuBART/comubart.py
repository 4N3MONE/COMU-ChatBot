from src.KoBART import get_kobart_model
import pytorch_lightning as pl

class ComuBART(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()
        self.model = get_kobart_model(model_path)
    
    def generate(self, inputs):
        result = self.model.generate(inputs,
                                     max_length=20,
                                     num_beams=2,
                                     top_p = 0.5,
                                     do_sample = True,
                                     no_repeat_ngram_size=2)
        
        return result
        
    def forward(self):
        pass