import json
from dataclasses import dataclass

@dataclass
class Config:
    n_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    gene_vocab_file: str
    train_data_file: str
    eval_data_file: str
    train_expression_file: str
    eval_expression_file: str
    output_dir: str
    max_length: int
    save_every: int
    expression_max_value: float
    expression_min_value: float
    num_bins: int
    device: str

# Function to read and load the JSON configuration file
def load_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return Config(**config_dict)