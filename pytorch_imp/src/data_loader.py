import os
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model_input_specification import ModelInputSpecification
from dataset_specification import DatasetSpecification, NamedDatasetSpecifications
from preprocess import StandardPreProcessing
from enumerations import CategoricalFormat  # 确保导入 CategoricalFormat
from preprocess import StandardPreProcessing  # 导入 StandardPreProcessing
from input_encodings import NoInputEncoder  # 导入 NoInputEncoder

import pdb

class WindowedSampler(Sampler):
    def __init__(self, data_source, window_size):
        """
        :param data_source: 数据集
        :param window_size: 滑动窗口大小
        """
        self.data_source = data_source
        self.window_size = window_size
        self.total_samples = len(data_source)

    def __iter__(self):
        """
        返回按 window_size 步长递增的索引
        """
        idx = 0
        while idx + self.window_size <= self.total_samples:
            yield idx
            idx += self.window_size
        
        # 如果剩余数据不足以填充一个窗口，可以选择返回最后一部分数据
        if idx < self.total_samples:
            yield idx

    def __len__(self):
        # 返回可以生成的总批次数
        return (self.total_samples // self.window_size) + (1 if self.total_samples % self.window_size != 0 else 0)

class CustomDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, specification, pre_processing, input_encoding, window_size=8):
        """
        初始化数据集
        """
        # 创建数据集的副本以避免修改原始数据
        self.dataset = dataset
        self.specification = specification
        self.pre_processing = pre_processing
        self.input_encoding = input_encoding
        self.window_size = window_size    
        self.numerical_filter = 1_000_000_000
        self.new_df = {}
        self.new_features = []
        self.model_input_spec = None

        # 获取需要处理的列
        self.numerical_columns = set(specification.include_fields).difference(specification.categorical_fields)
        self.categorical_columns = specification.categorical_fields
        
        # 对数据进行预处理
        self.preprocess_data()

    def preprocess_data(self):
        """ 预处理数据，标准化数值列、编码分类列 """
        # 数值列标准化
        for col_name in self.numerical_columns:
            assert col_name in self.dataset.columns
            # self.dataset.loc[col_name] = self.dataset[col_name]# .astype("float64")
            self.new_features.append(col_name)
            
            # 获取列值并进行预处理
            col_values = self.dataset.loc[:, col_name].values
            col_values[~np.isfinite(col_values)] = 0
            col_values[col_values < -self.numerical_filter] = 0
            col_values[col_values > self.numerical_filter] = 0
            # if == Nan -> set 0
            col_values[np.isnan(col_values)] = 0
            
            # 使用 StandardPreProcessing 对数值列进行预处理
            self.pre_processing.fit_numerical(col_name, col_values)
            col_values = self.pre_processing.transform_numerical(col_name, col_values)
            # self.dataset.loc[:, col_name] = col_values#.astype("float32")4
            self.new_df[col_name] = col_values
        
        # 分类列编码
        # pdb.set_trace()
        levels_per_categorical_feature = []
        for col_name in self.categorical_columns:
            if col_name == self.specification.class_column:
                continue
            # 使用 StandardPreProcessing 对分类列进行预处理
            self.new_features.append(col_name)
            col_values = self.dataset[col_name].values
            # if nan
            col_values[np.isnan(col_values)] = 0
            self.pre_processing.fit_categorical(col_name, col_values)
            new_values = self.pre_processing.transform_categorical(col_name, col_values, self.input_encoding.required_input_format)
            
            print(f"Encoding for {col_name}, new_values shape: {new_values.shape}, original data shape: {self.dataset.shape}")
            # pdb.set_trace()
            if self.input_encoding.required_input_format == CategoricalFormat.OneHot:
                # pdb.set_trace()
                # 对于 OneHot 编码，添加多个列
                if isinstance(new_values, pd.DataFrame):
                    # 如果是 DataFrame，则直接逐列添加到原始数据
                    levels_per_categorical_feature.append(len(new_values.columns))
                    for c in new_values.columns:
                        self.new_df[c] = new_values[c]
                else:
                    # 如果是 ndarray，则为每个类别创建新列
                    n_one_hot_levels = new_values.shape[1]
                    levels_per_categorical_feature.append(n_one_hot_levels)
                    for z in range(n_one_hot_levels):
                        new_col_name = f"{col_name}_{z}"
                        self.new_df[new_col_name] = new_values[:, z]
            else:
                # 对于整数编码
                if len(new_values) == len(self.dataset):  # 确保长度匹配
                    self.new_df[col_name] = new_values
                    levels_per_categorical_feature.append(len(np.unique(new_values)))
                else:
                    print(f"Warning: Length mismatch for column {col_name}. Expected {len(self.dataset)} but got {len(new_values)}.")
        
        self.new_df = pd.DataFrame(self.new_df)
        self.model_input_spec = ModelInputSpecification(self.new_features, len(self.numerical_columns), levels_per_categorical_feature, self.input_encoding.required_input_format)
        # pdb.set_trace()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取窗口的数据
        start_idx = idx
        end_idx = min(len(self.dataset), idx + self.window_size)

        # 获取特征数据
        x = self.new_df.iloc[start_idx:end_idx].values
        # print(f"x.shape: {x.shape}")  # 打印 x 的形状
        
        # 获取标签数据
        y_window = self.dataset[self.specification.class_column].iloc[start_idx:end_idx].values
        y = np.where(y_window == str(self.specification.benign_label), 0, 1)
        # print(f"y.shape: {y.shape}")  # 打印 y 的形状
        print("idx: ", idx)

        # 转换为tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

class DatasetLoader:
    def __init__(self, dataset: pd.DataFrame, specification, pre_processing, input_encoding, evaluation_dataset_sampling, evaluation_percent=0.2, batch_size=1, num_workers=1, drop_last=True):
        """
        初始化 DatasetLoader 类

        :param dataset: 预加载的数据集
        :param specification: 数据集规格
        :param pre_processing: 预处理器
        :param input_encoding: 输入编码
        :param evaluation_dataset_sampling: 评估数据集采样策略
        :param evaluation_percent: 评估数据集的百分比
        :param batch_size: 批量大小
        :param num_workers: 工作线程数
        """
        self.dataset = dataset
        self.specification = specification
        self.pre_processing = pre_processing
        self.input_encoding = input_encoding
        self.evaluation_dataset_sampling = evaluation_dataset_sampling
        self.evaluation_percent = evaluation_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 切分训练集和评估集
        self.train_dataset, self.eval_dataset = self.split_dataset()

    def split_dataset(self):
        """ 根据评估集比例切分数据集 """
        eval_n = int(len(self.dataset) * self.evaluation_percent)
        if self.evaluation_dataset_sampling == 'LastRows':
            # 取数据集最后 eval_n 行作为评估集
            eval_dataset = self.dataset[-eval_n:]
            train_dataset = self.dataset[:-eval_n]
        elif self.evaluation_dataset_sampling == 'RandomRows':
            # 随机抽取 eval_n 行作为评估集
            index = np.arange(len(self.dataset))
            sample = np.random.choice(index, eval_n, replace=False)
            eval_dataset = self.dataset.iloc[sample]
            train_dataset = self.dataset.drop(sample)
        
        return train_dataset, eval_dataset

    def load_data(self):
        """ 加载训练和评估数据集并返回 DataLoader """
        # 加载训练集 DataLoader
        train_custom_dataset = CustomDataset(self.train_dataset, self.specification, self.pre_processing, self.input_encoding)
        train_sampler = WindowedSampler(train_custom_dataset, window_size=8)
        train_loader = DataLoader(train_custom_dataset, batch_size=1, shuffle=False, sampler=train_sampler, num_workers=self.num_workers)
        
        # 加载评估集 DataLoader
        eval_custom_dataset = CustomDataset(self.eval_dataset, self.specification, self.pre_processing, self.input_encoding)
        eval_sampler = WindowedSampler(eval_custom_dataset, window_size=8)
        eval_loader = DataLoader(eval_custom_dataset, batch_size=1, shuffle=False, sampler=eval_sampler, num_workers=self.num_workers)

        # eval_loader = DataLoader(eval_custom_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return  train_loader, eval_loader

if __name__ == "__main__":
    # 假设你有一个数据集和规格说明
    dataset = pd.read_csv('../datasets.csv')  # 替换为实际路径
    specification = NamedDatasetSpecifications.unified_flow_format

    # preprocess method
    pre_processing = StandardPreProcessing(n_categorical_levels=32)
    # 使用 DatasetLoader 加载数据
    dataset_loader = DatasetLoader(dataset, specification, pre_processing=pre_processing, input_encoding=NoInputEncoder(), evaluation_dataset_sampling='LastRows', batch_size=64, num_workers=4)
    train_loader, eval_loader = dataset_loader.load_data()

    # 遍历训练集和评估集的批次

    print("Training Data:")
    for x, y in train_loader:
        print(x, y)

    print("Evaluation Data:")
    for x, y in eval_loader:
        print(x, y)