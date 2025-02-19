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
        values = values.reshape(N, value_len, self.heads, self.head_dim).permute(2, 0, 1, 3)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).permute(2, 0, 1, 3)
        query = query.reshape(N, query_len, self.heads, self.head_dim).permute(2, 0, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(query, keys.permute(0, 1, 3, 2))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, values)  # (N, heads, query_len, head_dim)
        out = out.permute(1, 2, 0, 3).contiguous().reshape(N, query_len, self.heads * self.head_dim)
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
        x = self.dropout(self.norm1(attention + query))  # Residual connection and layer normalization
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(self, model_args):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Linear(model_args.input_dim, model_args.embed_size)
        self.position_embedding = nn.Embedding(model_args.max_length, model_args.embed_size)

        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_args.embed_size, model_args.heads, model_args.dropout, model_args.forward_expansion) for _ in range(model_args.num_layers)]
        )

        self.fc_out = nn.Linear(model_args.embed_size, model_args.num_classes)  # Output layer for classification
        self.dropout = nn.Dropout(model_args.dropout)
        self.max_length = model_args.max_length

    def forward(self, src, src_mask=None):
        N = src.shape[0]
        src_len = src.shape[1]

        # Add position embeddings
        positions = torch.arange(0, src_len).unsqueeze(0).repeat(N, 1).to(src.device)

        # Adding word and position embeddings
        src = self.dropout(self.src_word_embedding(src) + self.position_embedding(positions))

        # Encoder
        for encoder in self.encoder_blocks:
            src = encoder(src, src, src, src_mask)

        # The output from the first token (CLS token) for classification
        cls_token_out = src[:, 0, :]  # Shape: (N, embed_size)

        # Classification head
        out = self.fc_out(cls_token_out)  # Shape: (N, num_classes)
        return out

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulated network traffic data (batch_size=32, sequence_length=100, feature_dim=64)
    input_data = torch.randn(32, 100, 64).to(device)  # Shape: (32, 100, 64)
    
    # Model hyperparameters
    input_dim = 64  # Each input feature vector has 64 elements
    embed_size = 128  # The size of the embedding dimension
    num_layers = 4  # The number of transformer encoder layers
    heads = 8  # Number of attention heads
    dropout = 0.1  # Dropout rate
    forward_expansion = 4  # Expansion factor for the feed-forward layer
    num_classes = 2  # Number of classes for classification (e.g., malicious or benign)
    
    # Instantiate the model
    model = Transformer(input_dim=input_dim, embed_size=embed_size, num_layers=num_layers, 
                        heads=heads, dropout=dropout, forward_expansion=forward_expansion, 
                        num_classes=num_classes).to(device)

    # Forward pass
    output = model(input_data)
    print(output.shape)  # Expected shape: (32, 2), i.e., batch_size x num_classes