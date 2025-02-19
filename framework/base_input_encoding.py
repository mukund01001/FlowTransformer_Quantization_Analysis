from typing import List
from framework.enumerations import CategoricalFormat
from framework.framework_component import FunctionalComponent
import torch
import torch.nn as nn


class BaseInputEncoding(FunctionalComponent):
    def apply(self, X: List["torch.Tensor"], prefix: str = None):
        raise NotImplementedError("Please override this with a custom implementation")

    @property
    def required_input_format(self) -> CategoricalFormat:
        raise NotImplementedError("Please override this with a custom implementation")


# # Example of a potential encoding class for integers or one-hot encodings
# class IntegerInputEncoding(BaseInputEncoding):
#     def __init__(self, embedding_dim: int = 8):
#         super().__init__()
#         self.embedding_dim = embedding_dim

#     def apply(self, X: List["torch.Tensor"], prefix: str = None):
#         # For example, assuming categorical features are integer encoded
#         return [torch.nn.functional.embedding(x, num_embeddings=10, embedding_dim=self.embedding_dim) for x in X]

#     @property
#     def required_input_format(self) -> CategoricalFormat:
#         return CategoricalFormat.Integers