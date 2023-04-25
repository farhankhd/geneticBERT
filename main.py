import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW, \
    BertConfig

from transformers import LongformerConfig, LongformerTokenizer, LongformerForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import scanpy as sc
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
import random


CLASS = 7

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


data_path = '/home/rohola/codes/geneticBERT/data/Zheng68K.h5ad'

# Set device to run training on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model and tokenizer
model_name = 'bert-base-cased'
# tokenizer = LongformerTokenizer.from_pretrained(model_name)

config = LongformerConfig.from_pretrained('allenai/longformer-base-4096',
                                          num_attention_heads=4,
                                          hidden_size=16,
                                          max_position_embeddings=17408,
                                          position_embedding_type='none')
config.vocab_size = CLASS
config.num_labels = 11

model = LongformerForSequenceClassification(config=config).to(device)


batch_size = 1

data = sc.read_h5ad(data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)

label = torch.from_numpy(label)
data = data.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Set optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                         num_training_steps=len(train_loader) * 10)

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        print(i)
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, val_acc = 0.0, 0.0
    for batch in val_loader:
        inputs, labels = batch
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
        val_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        val_acc += torch.sum(preds == labels).item()
    val_loss = val_loss / len(val_loader)
    val_acc = val_acc
