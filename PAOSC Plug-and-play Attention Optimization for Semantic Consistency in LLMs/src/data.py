import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from sklearn.utils import resample
from urllib.parse import unquote
from transformers import Qwen2Tokenizer


class DataCollatorForSeq2Seq():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features):
        encoded_input = []
        words = []
        #print(features)
        texts = [feature["content"] for feature in features]
        
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=False)

        batch_size, length = encoded_input['input_ids'].shape
        encoded_input['position_ids'] = torch.arange(0, length).unsqueeze(0).repeat(batch_size, 1)
        for input_ids in encoded_input['input_ids']:
            word = []
            for id in input_ids:
                word.append(self.tokenizer.decode(id))
            words.append(word)
        
        batch = {
            'input': encoded_input,
            'words':words,
            'label': [feature['label'] for feature in features],
            'content':texts,
        }
        return batch  


def get_dataloader(df:pd.core.frame.DataFrame, tokenizer:[Qwen2Tokenizer], test_size_ratio:float=0.2, batch_size:int=4):
    dataset = Dataset.from_pandas(df)
    train_dataset, test_dataset = dataset.train_test_split(test_size=test_size_ratio).values()
    collator = DataCollatorForSeq2Seq(tokenizer) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,collate_fn=collator)
    return train_loader, test_loader


def upsample(df:pd.core.frame.DataFrame):
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True,    
                                     n_samples=len(df_majority),  
                                     random_state=42)  
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_upsampled


def HTTP_RLDataLoader(tokenizer, data_path:str,  test_size_ratio:float=0.2, batch_size:int=4):
    df = pd.read_csv(data_path)
    df['content'] = df['url'].apply(unquote)
    df = upsample(df)
    train_loader, test_loader = get_dataloader(df=df,tokenizer=tokenizer,test_size_ratio=test_size_ratio,batch_size=batch_size)
    return train_loader, test_loader
    

def SpamDataLoader(tokenizer, data_path:str,  test_size_ratio:float=0.2, batch_size:int=4):
    df = pd.read_csv(data_path)
    df = upsample(df)
    train_loader, test_loader = get_dataloader(df=df,tokenizer=tokenizer,test_size_ratio=test_size_ratio,batch_size=batch_size)
    return train_loader, test_loader


def AGnewsDataLoader(tokenizer, data_path:str,  test_size_ratio:float=0.2, batch_size:int=4):
    df = pd.read_csv(data_path)
    df = df.rename(columns={'review': 'content','Class Index':'label'})
    df['label'] = df['label'].map({1: 0, 2: 1, 3: 2, 4: 3})
    train_loader, test_loader = get_dataloader(df=df,tokenizer=tokenizer,test_size_ratio=test_size_ratio,batch_size=batch_size)
    return train_loader, test_loader


def HttpPramaDataLoader(tokenizer, data_path:str,  test_size_ratio:float=0.2, batch_size:int=4):
    df = pd.read_csv(data_path)
    df['label'] = df['label'].map({'norm': 0, 'anom': 1})
    df = df.rename(columns={'payload': 'content'})
    df = upsample(df)
    train_loader, test_loader = get_dataloader(df=df,tokenizer=tokenizer,test_size_ratio=test_size_ratio,batch_size=batch_size)
    return train_loader, test_loader
    
    







    
