from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, rating):
        super(Dataset, self).__init__()
        self.user = rating['user_id']
        self.business = rating['item_id']
        self.rating = rating['rating']

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, item):
        return self.user[item], self.business[item], self.rating[item]


class ISKGTDDataset(Dataset):
    def __init__(self, rating):
        super(Dataset, self).__init__()
        self.user = rating['user_id']
        self.business = rating['business_id']
        self.rating = rating['stars']

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, item):
        return self.user[item], self.business[item], self.rating[item]