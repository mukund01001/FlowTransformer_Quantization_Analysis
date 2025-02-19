import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, BertModel, BertConfig

from framework.base_sequential import BaseSequential

class GPTSmallTransformer(BaseSequential):
    @property
    def name(self) -> str:
        return "GPT-2 Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size // self.n_heads
        self.dropout_rate = 0.02

        # Load the GPT-2 model configuration and model
        self.gpt2_config = GPT2Config(
            num_hidden_layers=self.n_layers,
            hidden_size=self.internal_size,
            num_attention_heads=self.n_heads,
            intermediate_size=self.internal_size * 4,
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate
        )
        self.gpt2_model = GPT2Model(self.gpt2_config)

    def forward(self, X):
        # Pass through the GPT-2 model
        gpt2_outputs = self.gpt2_model(input_ids=X)
        return gpt2_outputs.last_hidden_state  # We return the hidden states


class BERTSmallTransformer(BaseSequential):
    @property
    def name(self) -> str:
        return "BERT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size // self.n_heads
        self.dropout_rate = 0.02

        # Load the BERT model configuration and model
        self.bert_config = BertConfig(
            num_hidden_layers=self.n_layers,
            hidden_size=self.internal_size,
            num_attention_heads=self.n_heads,
            intermediate_size=self.internal_size * 4,
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate
        )
        self.bert_model = BertModel(self.bert_config)

    def forward(self, X):
        # Pass through the BERT model
        bert_outputs = self.bert_model(input_ids=X)
        return bert_outputs.last_hidden_state  # We return the hidden states