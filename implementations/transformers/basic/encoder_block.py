from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn

class GPT3Attention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(output)
        output = self.dropout(output)

        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv=False):
        super().__init__()

        self.attn = GPT3Attention(num_heads, input_dimension, dropout_rate)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_layer_norm = nn.LayerNorm(input_dimension)

        self.feed_forward_0 = nn.Conv1d(input_dimension, inner_dimension, kernel_size=1) if use_conv else nn.Linear(input_dimension, inner_dimension)
        self.feed_forward_1 = nn.Conv1d(inner_dimension, input_dimension, kernel_size=1) if use_conv else nn.Linear(inner_dimension, input_dimension)

        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_layer_norm = nn.LayerNorm(input_dimension)

    def forward(self, inputs, mask=None):
        x = inputs

        # Attention layer
        attention_output = self.attn(x, x, x, mask=mask)
        attention_output = self.attention_dropout(attention_output)
        x = x + attention_output
        x = self.attention_layer_norm(x)

        # Feedforward network
        feed_forward_output = self.feed_forward_0(x)
        feed_forward_output = self.feed_forward_1(feed_forward_output)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        x = x + feed_forward_output

        return self.feed_forward_layer_norm(x)