import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        # Linear layers for queries, keys, values, and output projection
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)


    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads different pieces 
        # alter reshape: (N, seq_len, embed_dim) -> (N, seq_len, heads, head_dim)
        # alter permute: (N, seq_len, heads, head_dim) -> (N, heads, seq_len, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim).permute(2, 0, 1, 3) 
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).permute(2, 0, 1, 3)
        query = query.reshape(N, query_len, self.heads, self.head_dim).permute(2, 0, 1, 3)

        # permute: (N, heads, seq_len, head_dim) -> (N, heads, seq_len, head_dim), easy to multiply with query
        # after matmul: (N, heads, query_len, key_len)
        energy = torch.matmul(query, keys.permute(0, 1, 3, 2))  # Scaled dot-product attention

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, values) # shape: (N, heads, query_len, head_dim)
        out = out.permute(1, 2, 0, 3).contiguous().reshape(N, query_len, self.heads * self.head_dim) # Reshape to (N, query_len, embed_dim)

        out = self.fc_out(out)
        return out
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query)) # Residual connection and layer normalization
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.encoder_attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, encoder_output):
        attention = self.attention(value, key, query, mask)
        query = self.dropout(self.norm1(attention + query))
        encoder_attention = self.encoder_attention(encoder_output, encoder_output, query, mask)
        query = self.dropout(self.norm2(encoder_attention + query))
        forward = self.feed_forward(query)
        out = self.dropout(self.norm3(forward + query))
        return out
    
class Transformer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embed_size, 
                 num_layers, 
                 heads, 
                 dropout, 
                 forward_expansion, 
                 max_length=512):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Linear(input_dim, embed_size)
        self.tgt_word_embedding = nn.Linear(input_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, src, tgt, src_mask, tgt_mask):
        N = src.shape[0]
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]

        # Add position embeddings
        src_positions = torch.arange(0, src_len).unsqueeze(0).repeat(N, 1).to(src.device)
        tgt_positions = torch.arange(0, tgt_len).unsqueeze(0).repeat(N, 1).to(tgt.device)

        src = self.dropout(self.src_word_embedding(src) + self.position_embedding(src_positions))
        tgt = self.dropout(self.tgt_word_embedding(tgt) + self.position_embedding(tgt_positions))

        # Encoder
        for encoder in self.encoder_blocks:
            src = encoder(src, src, src, src_mask)

        # Decoder
        for decoder in self.decoder_blocks:
            tgt = decoder(tgt, tgt, tgt, tgt_mask, src)

        out = self.fc_out(tgt)
        return out
    
if __name__ == "__main__":
    # Testing the transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0

    src_vocab_size = 10
    tgt_vocab_size = 10

    model = Transformer(src_vocab_size, tgt_vocab_size, embed_size=32, num_layers=2, heads=8, dropout=0.1, forward_expansion=4).to(device)
    #src_mask, trg_mask = model.create_masks(x, trg, src_pad_idx, trg_pad_idx)
    out = model(x, trg[:, :-1], src_mask=None, tgt_mask=None)
    print(out.shape)
