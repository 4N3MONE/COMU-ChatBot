import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,PreTrainedTokenizerFast



model_name = 'skt/ko-gpt-trinity-1.2B-v0.5'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
file_path = '../data/data_final.tsv'
data = pd.read_csv(file_path, sep = '\t')



U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained(model_name,
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

class ChatDataset(Dataset):
    def __init__(self, chats, max_len=30):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.select(list([idx]))
        q = turn['title']
        a = turn['comment']
        q_toked = self.tokenizer.tokenize(self.q_token + str(q))                                   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + str(a) + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)
        
        
