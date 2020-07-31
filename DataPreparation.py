# Loading required modules
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class disaster_data(Dataset):
    
    def __init__(self, dataset, encoder):
        super(disaster_data, self).__init__()
        self.encoder = encoder
        self.data = dataset
        self.text = [row['text'] for row in self.data]
        self.labels = [row['target'] for row in self.data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        holder = {}
        encoded = self.encoder.batch_encode_plus([self.text[index]], max_length = 30, truncation = True, pad_to_max_length = True)
        holder['embedding'] = torch.tensor(encoded['input_ids']).squeeze()
        holder['mask'] = torch.tensor(encoded['attention_mask']).squeeze()
        holder['label'] = float(self.labels[index])
        return holder

class Disaster_test_set(Dataset):
    
    def __init__(self, dataset, encoder):
        super(Disaster_test_set, self).__init__()
        self.encoder = encoder
        self.data = dataset
        self.text = [text for text in self.data['text']]
        self.id = dataset['id']
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        holder = {}
        encoded = self.encoder.batch_encode_plus([self.text[index]], max_length = 30, truncation = True, pad_to_max_length = True)
        holder['id'] = self.id[index]
        holder['embedding'] = torch.tensor(encoded['input_ids']).squeeze()
        holder['mask'] = torch.tensor(encoded['attention_mask']).squeeze()
        return holder

def remove_url(text):
    compiler = re.compile(r'https?://\S+|www\.\S+')
    
    output = compiler.sub('', text)
    
    return output

def remove_html(text):
    compiler = re.compile(r'<.*?>')
    
    output = compiler.sub('', text)
    
    return output

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    compiler = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    output = compiler.sub('', text)
    
    return output

def remove_hashtag(text):
    compiler = re.compile(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
    output = compiler.sub('', text)
    
    return output

def data_cleaning(text):
    text = remove_url(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_hashtag(text)
    
    return text

def prepare_training_data(cleaning = False, test_size = 0.2):
    '''
    Providing one way data preparation flow

    Args:
        cleaning (Bool): Whether to do data preprocessing procedure

    Returns:
        [Dict...] Every element is the list contains a dictionay representing every single row
    '''
    train_pd = pd.read_csv('train.csv')
    train_text, val_text, train_label, val_label = train_test_split(train_pd['text'], train_pd['target'],
                                                                    test_size = test_size, random_state = 2020)
    
    clean = data_cleaning if cleaning else lambda x : x
    train_data = [{'text' : clean(text), 'target' : label} for text, label in zip(train_text, train_label)]
    val_data = [{'text' : clean(text), 'target' : label} for text, label in zip(val_text, val_label)]  

    return train_data, val_data

def get_dl_training_set(encoder, batch_size):
    '''
    Creating data loader

    Args:
        encoder: the tokenizer for raw text
        batch_size(int): The number of samples in each batch
    
    Returns:
        trainLoader: The pytorch training data loader
        valLoader: The pytorch validation data loader
    '''
    train_data, val_data = prepare_training_data()
    trainSet = disaster_data(train_data, encoder)
    valSet = disaster_data(val_data, encoder)
    trainLoader = DataLoader(trainSet, batch_size = batch_size, shuffle = True)
    valLoader = DataLoader(valSet, batch_size = batch_size, shuffle = False)
    
    return trainLoader, valLoader

def get_dl_testing_set(encoder):
    '''
    Creating data loader

    Args:
        encoder: the tokenizer for raw text
        batch_size(int): The number of samples in each batch
    
    Returns:
        testLoader: The pytorch testing data loader
    '''
    test_pd = pd.read_csv('test.csv')
    testSet = Disaster_test_set(test_pd, encoder)
    testLoader = DataLoader(testSet, batch_size = 32, shuffle = False)

    return testLoader