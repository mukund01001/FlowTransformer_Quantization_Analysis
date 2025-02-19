import torch
import torch.nn as nn
import numpy as np

from framework.base_classification_head import BaseClassificationHead

class FlattenClassificationHead(BaseClassificationHead):
    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""
        return X.view(X.size(0), -1)  # Flatten the tensor

    @property
    def name(self) -> str:
        return "Flatten"

    @property
    def parameters(self) -> dict:
        return {}


class FeaturewiseEmbedding(BaseClassificationHead):
    def __init__(self, project: bool = False):
        super().__init__()
        self.project: bool = project

    @property
    def name(self):
        if self.project:
            return "Featurewise Embed - Projection"
        else:
            return "Featurewise Embed - Dense"

    @property
    def parameters(self):
        return {}

    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""

        if self.model_input_specification is None:
            raise Exception("Please call build() before calling apply!")

        # Linear layer (Dense) and projection logic
        linear = nn.Linear(X.size(-1), 1)  # Apply projection to the feature
        x = linear(X)

        # Flatten the output to match the original behavior
        return x.view(x.size(0), -1)


class GlobalAveragePoolingClassificationHead(BaseClassificationHead):
    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""
        
        # Use AdaptiveAvgPool1d for Global Average Pooling
        pool = nn.AdaptiveAvgPool1d(1)
        X = X.transpose(1, 2)  # Transpose for pooling: (batch_size, channels, seq_length)
        return pool(X).squeeze(-1)  # Remove the last dimension (seq_length)


    @property
    def name(self) -> str:
        return "Global Average Pooling"

    @property
    def parameters(self) -> dict:
        return {}


class LastTokenClassificationHead(BaseClassificationHead):
    def __init__(self):
        super().__init__()

    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""

        # Get the last token (i.e., the last element in the sequence dimension)
        return X[:, -1, :]  # Select the last token along the sequence dimension

    @property
    def name(self) -> str:
        return "Last Token"

    @property
    def parameters(self) -> dict:
        return {}


class CLSTokenClassificationHead(LastTokenClassificationHead):
    @property
    def name(self) -> str:
        return "CLS Token"

    @property
    def parameters(self) -> dict:
        return {}

    def apply_before_transformer(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""

        batch_size = X.size(0)
        window_size = X.size(1)
        flow_size = X.size(2)

        # Create the CLS token and append it to the input tensor (similar to BERT)
        cls_token = torch.ones(batch_size, 1, flow_size).to(X.device)  # CLS token for batch

        # Concatenate CLS token vertically
        X = torch.cat([cls_token, X], dim=1)

        # Generate horizontal CLS token for each sequence element
        cls_token_horizontal = torch.ones(batch_size, window_size + 1, 1).to(X.device)

        # Concatenate horizontal CLS token
        X = torch.cat([X, cls_token_horizontal], dim=-1)  # Concatenate in the feature dimension

        return X