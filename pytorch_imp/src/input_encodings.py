import warnings
import torch
import torch.nn as nn
from typing import List

from base.base_input_encoding import BaseInputEncoding
from src.enumerations import CategoricalFormat

from enum import Enum

class NoInputEncoder(BaseInputEncoding):
    def apply(self, X, prefix: str = None):
        # Assuming X is a list of PyTorch tensors
        numerical_feature_inputs = X[:self.model_input_specification.n_numeric_features]
        categorical_feature_inputs = X[self.model_input_specification.n_numeric_features:]

        if self.model_input_specification.categorical_format == CategoricalFormat.Integers:
            warnings.warn("It doesn't make sense to be using integer-based inputs without encoding!")
            categorical_feature_inputs = [x.float() for x in categorical_feature_inputs]  # Convert to float

        concat = torch.cat(numerical_feature_inputs + categorical_feature_inputs, dim=-1)

        return concat

    @property
    def name(self):
        return "No Input Encoding"

    @property
    def parameters(self):
        return {}

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.OneHot


class EmbedLayerType(Enum):
    Dense = 0,
    Lookup = 1,
    Projection = 2


class RecordLevelEmbed(BaseInputEncoding):
    def __init__(self, embed_dimension: int, project: bool = False):
        super().__init__()

        self.embed_dimension: int = embed_dimension
        self.project: bool = project

    @property
    def name(self):
        if self.project:
            return "Record Level Projection"
        return "Record Level Embedding"

    @property
    def parameters(self):
        return {
            "dimensions_per_feature": self.embed_dimension
        }

    def apply(self, X: List[torch.Tensor], prefix: str = None):
        if prefix is None:
            prefix = ""

        assert self.model_input_specification.categorical_format == CategoricalFormat.OneHot

        x = torch.cat(X, dim=-1)
        linear = nn.Linear(x.size(-1), self.embed_dimension)
        x = linear(x)

        return x

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.OneHot


class CategoricalFeatureEmbed(BaseInputEncoding):
    def __init__(self, embed_layer_type: EmbedLayerType, dimensions_per_feature: int):
        super().__init__()

        self.dimensions_per_feature: int = dimensions_per_feature
        self.embed_layer_type: EmbedLayerType = embed_layer_type

    @property
    def name(self):
        if self.embed_layer_type == EmbedLayerType.Dense:
            return "Categorical Feature Embed - Dense"
        elif self.embed_layer_type == EmbedLayerType.Lookup:
            return "Categorical Feature Embed - Lookup"
        elif self.embed_layer_type == EmbedLayerType.Projection:
            return "Categorical Feature Embed - Projection"
        raise RuntimeError()

    @property
    def parameters(self):
        return {
            "dimensions_per_feature": self.dimensions_per_feature
        }

    def apply(self, X: List[torch.Tensor], prefix: str = None):
        if prefix is None:
            prefix = ""

        if self.model_input_specification is None:
            raise Exception("Please call build() before calling apply!")

        numerical_feature_inputs = X[:self.model_input_specification.n_numeric_features]
        categorical_feature_inputs = X[self.model_input_specification.n_numeric_features:]

        collected_numeric = torch.cat(numerical_feature_inputs, dim=-1)

        collected_categorical = []
        for categorical_field_i, categorical_field_name in enumerate(self.model_input_specification.categorical_feature_names):
            cat_field_x = categorical_feature_inputs[categorical_field_i]
            if self.embed_layer_type != EmbedLayerType.Lookup:
                assert self.model_input_specification.categorical_format == CategoricalFormat.OneHot

                linear = nn.Linear(cat_field_x.size(-1), self.dimensions_per_feature)
                x = linear(cat_field_x)
                collected_categorical.append(x)

            elif self.embed_layer_type == EmbedLayerType.Lookup:
                assert self.model_input_specification.categorical_format == CategoricalFormat.Integers

                embedding = nn.Embedding(self.model_input_specification.levels_per_categorical_feature[categorical_field_i] + 1, self.dimensions_per_feature)
                x = embedding(cat_field_x)
                collected_categorical.append(x)

        collected_categorical = torch.cat(collected_categorical, dim=-1)

        collected = torch.cat([collected_numeric, collected_categorical], dim=-1)

        return collected

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.Integers if self.embed_layer_type == EmbedLayerType.Lookup else CategoricalFormat.OneHot