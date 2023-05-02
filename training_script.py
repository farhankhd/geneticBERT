from torch.utils.data import Dataset, DataLoader
import torch
from transformers import DataCollatorForLanguageModeling, BertConfig, BertForMaskedLM
from transformers import PreTrainedTokenizer
import torch.optim as optim


class GeneticDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_len):
        super().__init__()
        self.data = [item.rstrip() for item in open(data_path, 'r').readlines()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # read special tokens
        pad_token_id = self.tokenizer.pad_token_id
        # convert the data to a list of indexes
        # input_ids = [8, 7, 3, 6]
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(data))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = torch.nn.functional.pad(input_ids, (0, self.max_len - len(input_ids)), value=pad_token_id)
        # create the attention mask which is 1 for all non padding tokens and 0 for all padding tokens
        attention_mask = [1 if x != pad_token_id else 0 for x in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class GeneTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, pad_token="<PAD>", mask_token="<MASK>", **kwargs):
        super().__init__(pad_token=pad_token, mask_token=mask_token, **kwargs)
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.vocab = [pad_token, mask_token, "<UNK>"] + vocab
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}

    def _tokenize(self, text):
        return text.split(', ')

    def _convert_token_to_id(self, token):
        return self.token2idx.get(token, self.token2idx["<UNK>"])

    def _convert_id_to_token(self, index):
        return self.idx2token.get(index, "<UNK>")

    def convert_tokens_to_string(self, tokens):
        return ', '.join(tokens)

    def tokenize(self, text, **kwargs):
        return self._tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(index) for index in ids]

    @property
    def pad_token_id(self):
        return self.token2idx[self.pad_token]

    @property
    def mask_token_id(self):
        return self.token2idx[self.mask_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# sample vocab

vocab = [item.rstrip() for item in open("gene_data/gene_vocab.txt").readlines()]
tokenizer = GeneTokenizer(vocab)

# input = 'ABHD2, ANKFN1, APOC4-APOC2'
#
# tokenizer = GeneTokenizer(vocab)
# x = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
# print(x)
# print(tokenizer.tokenize(input))
# print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input)))

sample_input_path = "gene_data/train.txt"
dataset = GeneticDataset(sample_input_path, tokenizer, 512)
# for sample in dataset:
#     print(sample)


# Set up the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Set up the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=data_collator
)

# Set up the model
config = BertConfig(vocab_size=len(tokenizer.vocab))
model = BertForMaskedLM(config=config).to('cpu')

# Set up the optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    model.train()
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        print(f"Batch {i + 1} - Loss: {loss.item()}")
