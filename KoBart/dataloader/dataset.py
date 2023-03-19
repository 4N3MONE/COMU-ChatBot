from torch.utils.data import Dataset
import os
from load_model import get_kobart_tokenizer

class KoBARTDataset(Dataset):
    def __init__(self,
                 file_path = os.path.join(os.getcwd() + "/../data/dataset.tsv"),
                 n_ctx =1024,
                 ignore_header=True):
        self.file_path = file_path
        self.data = []
        self.tokenizer = get_kobart_tokenizer()

        bos_token_id = [self.tokenizer.bos_token_id]
        eos_token_id = [self.tokenizer.eos_token_id]
        pad_token_id = [self.tokenizer.pad_token_id]

        file = open(self.file_path, 'r', encoding='utf-8')
        first_line = True
        print_line = 5

        while True:
            line = file.readline()
            if ignore_header and first_line:
                first_line = False
                continue
            if not line:
                break
            
            data = line.split('\t')
            if print_line > 0:
                print_line -= 1
                print(data)
            index_of_words = bos_token_id + self.tokenizer.encode(data[1]) + eos_token_id + eos_token_id + self.tokenizer.encode(data[2]) + eos_token_id
            pad_token_len = n_ctx - len(index_of_words)

            index_of_words += pad_token_id * pad_token_len

            self.data.append(index_of_words)
        
        file.close()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item
    
if __name__=='__main__':
    dataset = KoBARTDataset()
    print(dataset[0])