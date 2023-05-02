import torch
from transformers import BertForMaskedLM, BertConfig, DataCollatorForLanguageModeling, BertTokenizer

config = BertConfig(vocab_size=5600)
model = BertForMaskedLM(config=config).to('cpu')

# print(model.get_input_embeddings().weight)

max_seq_length = model.config.max_position_embeddings
PAD = 0
a = [5, 6, 8, 100, 231]
input_ids = torch.Tensor(a)
input_ids = torch.nn.functional.pad(input_ids, (0, max_seq_length - input_ids.shape[-1]), value=PAD)

input_ids = input_ids.unsqueeze(0).long()

output = model(input_ids, output_hidden_states=True)

# the hidden states of the last layer
print(output['hidden_states'][-1])

print(output[0].shape)

# pass the masked input to the model

MASK = 1
inputs = [5, 6, 8, 23, 231]
masked_label = [5, 6, 8, MASK, 231]

input_ids = torch.Tensor(inputs)
input_ids = torch.nn.functional.pad(input_ids, (0, max_seq_length - input_ids.shape[-1]), value=PAD)
input_ids = input_ids.unsqueeze(0).long()

mask_token_index = masked_label.index(MASK)
labels = torch.zeros_like(input_ids)
labels[0, mask_token_index] = 100

outputs = model(input_ids, labels=labels)
loss = outputs.loss
print(loss.item())



#####
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

mask_labels = torch.ones_like(input_ids)
data_collator.mask_tokens(input_ids, mask_labels)


