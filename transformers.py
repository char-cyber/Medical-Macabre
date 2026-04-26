import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import string
import re
from tqdm import tqdm
import math
#cleaning code 
def preprocess_text(text):
    #tokenizing clinical notes 
    if isinstance(text, str):
        text = text.lower()# convert text to lower case
        text = re.sub(f'[{string.punctuation}]', '', text)#remove any punctuation 
        text = re.sub(r'\d+', '', text)#remove number to just focus on imp keywords??
        return text.split() # split text into a string of words 
    return []

class Vocabulary:
    #maps medical words to unique integers 
    def __init__(self, max_size):
        self.max_size = max_size
        # these indices are reserved for padding, unknown words and classification 
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size = 3

    def add_word(self, word):
        #this tracks how many times each word appears to just filter the most important ones  
        self.word_count[word] = self.word_count.get(word, 0) + 1

    def build_vocab(self):
        #build a vocab map based on how frequent a word is found in the dataset while training 
        #takes unique items and stores them from most freq to least 
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        #map words to numbers 
        for word, _ in sorted_words[:self.max_size - self.size]:
            self.word2idx[word] = self.size
            self.idx2word[self.size] = word
            self.size += 1

    def text_to_indices(self, tokens, max_len):
        #convert a list of words into a fixed length list of integers 
        indices = [self.word2idx["<cls>"]]#start every sequence with a class token
        for token in tokens:
            #map word to index or use unk if word is not in our vocab 
            indices.append(self.word2idx.get(token, self.word2idx["<unk>"]))
        indices = indices[:max_len]#shorten notes to be less than 128 words 
        if len(indices) < max_len:
            # add pad token to notes shorter than 128 word limit
            
            indices += [self.word2idx["<pad>"]] * (max_len - len(indices))
        return indices

class MedicalDataset(Dataset):
#dataset to feed clinical text and labels into the model 
    def __init__(self, dataframe, vocabulary, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.vocab = vocabulary
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = preprocess_text(row['text'])
        indices = self.vocab.text_to_indices(tokens, self.max_len)
        input_tensor = torch.tensor(indices, dtype=torch.long)
        
        #if label is 1 its usable else its not
        label = 0
        if 'label' in self.data.columns:
            val = str(row['label']).lower()
            label = 1 if val in ['1', 'yes', 'useful'] else 0
        
        label_tensor = torch.tensor([label], dtype=torch.float)
        #ignorring padding during self attention calculations 
        attention_mask = (input_tensor != self.vocab.word2idx["<pad>"]).long()
        return input_tensor, attention_mask, label_tensor

class PositionalEncoding(nn.Module):
    #add position information into transformers since they process information in parallel 
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    #Analyze relation between words in a node 
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, ff_dim=256, max_len=128, dropout=0.1):
        super().__init__()
        # converts interger IDs into 1
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.pos_enc(self.embedding(input_ids))
        key_padding_mask = (attention_mask == 0)
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        
        # Global Average Pooling across non-pad tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.fc(self.dropout(pooled))

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # Set model to training mode (enables dropout)
    for inputs, mask, labels in tqdm(loader, desc="Training"):
        inputs, mask, labels = inputs.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, mask)# Get predictions
        loss = criterion(outputs, labels)# Calculate how wrong the predictions were
        loss.backward()# Calculate gradients (backpropagation)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Use CSV files per project instructions 
    TRAIN_FILE = "train_data-text_and_labels.csv"
    TEST_FILE = "testX_text_only.csv" # Update to match your specific test file name
    MAX_LEN = 128 #project constraints 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_train = pd.read_csv(TRAIN_FILE)
    
    # Initialize vocabulary with a maximum of 10,000 unique medical words (maybe)
    vocab = Vocabulary(max_size=10000)
    for text in df_train['text']:
        for token in preprocess_text(text):
            vocab.add_word(token)
    vocab.build_vocab()
    # Load data in batches of 16 for memory efficiency
    train_loader = DataLoader(MedicalDataset(df_train, vocab, MAX_LEN), batch_size=16, shuffle=True)
    model = TransformerEncoder(vocab_size=vocab.size, max_len=MAX_LEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # BCEWithLogitsLoss is ideal for Binary Classification (Yes/No)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        #save both weights and vocab so the model can be used offline 
    torch.save({'model_state': model.state_dict(), 'vocab': vocab}, "transformer_medical.pt")

if __name__ == "__main__":
    main()