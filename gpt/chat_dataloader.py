import torch
from torch.utils.data import DataLoader
from chat_dataset import ChatDataset

class ChatDataLoader(DataLoader):
    def __init__(self,dataset,batch_size, shuffle=True):
        super(ChatDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=self.collate_fn
        )
    # def collate_fn(self,batch):
    #     data = [item[0] for item in batch]
    #     mask = [item[1] for item in batch]
    #     label = [item[2] for item in batch]
    #     return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)