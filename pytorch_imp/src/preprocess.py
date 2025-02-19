#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import numpy as np
import pandas as pd
import os
import sys

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

from base.base_preprocessing import BasePreProcessing
from src.enumerations import CategoricalFormat


class StandardPreProcessing(BasePreProcessing):
    def __init__(self, n_categorical_levels: int, clip_numerical_values:bool=False):
        super().__init__()
        self.n_categorical_levels:int = n_categorical_levels
        self.clip_numerical_values:bool = clip_numerical_values
        self.min_range = {}
        self.encoded_levels = {}

    @property
    def name(self) -> str:
        return "Standard Preprocessing"

    @property
    def parameters(self) -> dict:
        return {
            "n_categorical_levels": self.n_categorical_levels,
            "clip_numerical_values": self.clip_numerical_values
        }

    def fit_numerical(self, column_name: str, values: np.array):

        v0 = np.min(values)
        v1 = np.max(values)
        r = v1 - v0

        self.min_range[column_name] = (v0, r)

    def transform_numerical(self, column_name: str, values: np.array):
        col_min, col_range = self.min_range[column_name]

        if col_range == 0:
            return np.zeros_like(values, dtype="float32")

        # center on zero
        values -= col_min

        # apply a logarithm
        col_values = np.log(values + 1)

        # scale max to 1
        col_values *= 1. / np.log(col_range + 1)

        if self.clip_numerical_values:
            col_values = np.clip(col_values, 0., 1.)

        return col_values

    def fit_categorical(self, column_name: str, values: np.array):
        levels, level_counts = np.unique(values, return_counts=True)
        sorted_levels = list(sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True))
        self.encoded_levels[column_name] = [s[0] for s in sorted_levels[:self.n_categorical_levels]]


    # def transform_categorical(self, column_name:str, values: np.array, expected_categorical_format: CategoricalFormat):
    #     encoded_levels = self.encoded_levels[column_name]
    #     print(f"Encoding the {len(encoded_levels)} levels for {column_name}")

    #     result_values = np.ones(len(values), dtype="uint32")
    #     for level_i, level in enumerate(encoded_levels):
    #         level_mask = values == level

    #         # we use +1 here, as 0 = previously unseen, and 1 to (n + 1) are the encoded levels
    #         result_values[level_mask] = level_i + 1

    #     if expected_categorical_format == CategoricalFormat.Integers:
    #         return result_values

    #     v = pd.get_dummies(result_values, prefix=column_name)
    #     return v
    def transform_categorical(self, column_name: str, values: np.array, expected_categorical_format: CategoricalFormat):
        encoded_levels = self.encoded_levels[column_name]
        print(f"Encoding the {len(encoded_levels)} levels for {column_name}")

        # 直接创建一个形状为 (len(values), num_levels) 的零矩阵
        num_levels = len(encoded_levels)
        result_values = np.zeros((len(values), num_levels), dtype=np.float32)  # 创建全零矩阵，dtype 为 float32

        # 遍历每个类别的 level，将对应的列设置为 1
        for level_i, level in enumerate(encoded_levels):
            level_mask = values == level
            result_values[level_mask, level_i] = 1  # 将对应的索引位置设置为 1

        if expected_categorical_format == CategoricalFormat.Integers:
            # 如果期望整数编码，返回整数编码
            return result_values.argmax(axis=1)  # 返回每行最大值的索引作为整数编码

        # 如果期望 One-Hot 编码，返回 One-Hot 矩阵
        return result_values





