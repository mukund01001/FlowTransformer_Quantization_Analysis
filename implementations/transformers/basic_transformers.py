import torch
import torch.nn as nn
from transformers import GPT2Model, BertModel, BertConfig

from framework.base_sequential import BaseSequential
from implementations.transformers.basic.decoder_block import TransformerDecoderBlock
from implementations.transformers.basic.encoder_block import TransformerEncoderBlock

class BasicTransformer(BaseSequential):
    @property
    def name(self) -> str:
        if self.use_conv:
            return f"Basic Conv Transformer" + (" Decoder" if self.is_decoder else "")
        else:
            return f"Basic Dense Transformer" + (" Decoder" if self.is_decoder else "")

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "use_conv": self.use_conv,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.internal_size
        }

    def __init__(self, n_layers:int, internal_size:int, n_heads:int, use_conv:bool=False, dropout_rate:float=0.1, is_decoder=False):
        super().__init__()
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.use_conv = use_conv
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.is_decoder = is_decoder

        # Define the decoder and encoder blocks using PyTorch equivalents
        self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(internal_size, internal_size, n_heads, dropout_rate) for _ in range(n_layers)]) if is_decoder else nn.ModuleList([TransformerEncoderBlock(internal_size, internal_size, n_heads, dropout_rate, use_conv) for _ in range(n_layers)])

    def forward(self, X):
        real_size = X.size(-1)

        m_x = X

        for layer_i in range(self.n_layers):
            if self.is_decoder:
                m_x = self.decoder_blocks[layer_i](m_x, m_x)  # Decoder block applies self-attention
            else:
                m_x = self.encoder_blocks[layer_i](m_x)  # Encoder block applies self-attention

        return m_x