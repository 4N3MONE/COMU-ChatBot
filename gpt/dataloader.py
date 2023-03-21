import pandas as pd
import torch.utils.data as data

class CustomDataset(data.Dataset):
    def __init__(self, file_path): # where we basically tokenize and store the data
        self.data = pd.read_csv(file_path, sep='\t', header=None, names=['input', 'label'])
    
    def __len__(self): # where we return the length of the total dataset,required for step size calculation within each epoch
        return len(self.data)
    
    def __getitem__(self, index): #where we fetch one data and then return it
        title_data = self.data.iloc[index]['title']
        comment_data = self.data.iloc[index]['comment']
        return title_data, comment_data

