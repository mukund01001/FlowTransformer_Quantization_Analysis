from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1):
        super().__init__()

        # Multi-head Attention Layer from Hugging Face's transformers
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dimension, num_heads=num_heads, dropout=dropout_rate)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dimension, inner_dimension),
            nn.ReLU(),
            nn.Linear(inner_dimension, input_dimension)
        )

        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(input_dimension)
        self.layer_norm2 = nn.LayerNorm(input_dimension)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, target_seq, enc_output):
        # Self-attention of target sequence
        attn_output, _ = self.multi_head_attention(target_seq, target_seq, target_seq)
        attn_output = self.dropout1(attn_output)
        out1 = target_seq + attn_output
        out1 = self.layer_norm1(out1)

        # Cross-attention with encoder output as the key and value
        attn_output, _ = self.multi_head_attention(out1, enc_output, enc_output)
        attn_output = self.dropout2(attn_output)
        out2 = out1 + attn_output
        out2 = self.layer_norm2(out2)

        # Feedforward network
        ffn_output = self.ffn(out2)
        out3 = out2 + ffn_output
        out3 = self.layer_norm2(out3)

        return out3