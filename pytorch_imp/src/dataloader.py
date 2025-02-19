import os
import warnings
import pandas as pd
import numpy as np

from typing import Union, Optional, Tuple, List
from model_input_specification import ModelInputSpecification
from utils import get_identifier, save_feather_plus_metadata, load_feather_plus_metadata
from enumerations import EvaluationDatasetSampling, CategoricalFormat

class DatasetLoader:
    def __init__(self, pre_processing, input_encoding, rs, retain_inmem_cache=False, inmem_cache=None):
        """
        初始化 DatasetLoader 类实例

        :param pre_processing: 数据预处理器
        :param input_encoding: 输入编码处理器
        :param rs: 随机数生成器
        :param retain_inmem_cache: 是否保留内存中的缓存
        :param inmem_cache: 内存缓存
        """
        self.pre_processing = pre_processing
        self.input_encoding = input_encoding
        self.rs = rs if rs is not None else np.random.RandomState()
        self.retain_inmem_cache = retain_inmem_cache
        self.inmem_cache = inmem_cache if retain_inmem_cache else None
        self.dataset_specification = None
        self.X = None
        self.y = None
        self.training_mask = None
        self.model_input_spec = None

    def _load_preprocessed_dataset(self, dataset_name: str,
                                   dataset: Union[pd.DataFrame, str],
                                   specification,  # Assume DatasetSpecification type
                                   cache_folder: Optional[str] = None,
                                   n_rows: int = 0,
                                   evaluation_dataset_sampling=None,  # Assume EvaluationDatasetSampling type
                                   evaluation_percent: float = 0.2,
                                   numerical_filter: int = 1_000_000_000) -> Tuple[pd.DataFrame, 'ModelInputSpecification']:
        """
        内部方法：加载和预处理数据集

        :param dataset_name: 数据集名称
        :param dataset: 数据集，可以是 DataFrame 或路径
        :param specification: 数据集的规格说明
        :param cache_folder: 缓存目录
        :param n_rows: 数据集的行数
        :param evaluation_dataset_sampling: 评估数据集采样策略
        :param evaluation_percent: 评估数据集的百分比
        :param numerical_filter: 数值过滤器
        :return: 处理后的数据集和模型输入规格
        """
        cache_file_path = None

        if dataset_name is None:
            raise Exception("Dataset name must be specified so FlowTransformer can optimise operations between subsequent calls!")

        pp_key = get_identifier({
            "__preprocessing_name": self.pre_processing.name,
            **self.pre_processing.parameters
        })

        local_key = get_identifier({
            "evaluation_percent": evaluation_percent,
            "numerical_filter": numerical_filter,
            "categorical_method": str(self.input_encoding.required_input_format),
            "n_rows": n_rows,
        })

        cache_key = f"{dataset_name}_{n_rows}_{pp_key}_{local_key}"

        if self.retain_inmem_cache:
            if self.inmem_cache is not None and cache_key in self.inmem_cache:
                print("Using in-memory cached version of this pre-processed dataset.")
                return self.inmem_cache[cache_key]

        if cache_folder is not None:
            cache_file_name = f"{cache_key}.feather"
            cache_file_path = os.path.join(cache_folder, cache_file_name)

            print(f"Using cache file path: {cache_file_path}")

            if os.path.exists(cache_file_path):
                print(f"Reading directly from cache {cache_file_path}...")
                return load_feather_plus_metadata(cache_file_path)

        if isinstance(dataset, str):
            print(f"Attempting to read dataset from path {dataset}...")
            if dataset.lower().endswith(".feather"):
                dataset = pd.read_feather(dataset, columns=specification.include_fields + [specification.class_column])
            elif dataset.lower().endswith(".csv"):
                dataset = pd.read_csv(dataset, nrows=n_rows if n_rows > 0 else None)
            else:
                raise Exception("Unrecognised dataset filetype!")
        elif not isinstance(dataset, pd.DataFrame):
            raise Exception("Unrecognised dataset input type, should be a path to a CSV or feather file, or a pandas dataframe!")

        assert isinstance(dataset, pd.DataFrame)

        if 0 < n_rows < len(dataset):
            dataset = dataset.iloc[:n_rows]

        training_mask = np.ones(len(dataset), dtype=bool)
        eval_n = int(len(dataset) * evaluation_percent)

        # Handle evaluation dataset sampling logic
        if evaluation_dataset_sampling == EvaluationDatasetSampling.LastRows:
            training_mask[-eval_n:] = False
        elif evaluation_dataset_sampling == EvaluationDatasetSampling.RandomRows:
            index = np.arange(self.parameters.window_size, len(dataset))
            sample = self.rs.choice(index, eval_n, replace=False)
            training_mask[sample] = False
        elif evaluation_dataset_sampling == EvaluationDatasetSampling.FilterColumn:
            training_column = dataset.columns[-1]
            print(f"Using the last column {training_column} as the training mask column")
            v, c = np.unique(dataset[training_column].values, return_counts=True)
            min_index = np.argmin(c)
            min_v = v[min_index]
            eval_indices = np.argwhere(dataset[training_column].values == min_v).reshape(-1)
            eval_indices = eval_indices[(eval_indices > self.parameters.window_size)]
            training_mask[eval_indices] = False
            del dataset[training_column]

        # Process numerical columns
        numerical_columns = set(specification.include_fields).difference(specification.categorical_fields)
        categorical_columns = specification.categorical_fields
        new_df = {"__training": training_mask, "__y": dataset[specification.class_column].values}

        print("Converting numerical columns to floats and removing out of range values...")
        for col_name in numerical_columns:
            assert col_name in dataset.columns
            col_values = dataset[col_name].values
            col_values[~np.isfinite(col_values)] = 0
            col_values[col_values < -numerical_filter] = 0
            col_values[col_values > numerical_filter] = 0
            new_df[col_name] = col_values.astype("float32")

        print(f"Applying pre-processing to numerical values")
        for i, col_name in enumerate(numerical_columns):
            all_data = new_df[col_name]
            training_data = all_data[training_mask]
            self.pre_processing.fit_numerical(col_name, training_data)
            new_df[col_name] = self.pre_processing.transform_numerical(col_name, all_data)

        print(f"Applying pre-processing to categorical values")
        levels_per_categorical_feature = []
        new_features = []
        
        for i, col_name in enumerate(categorical_columns):
            new_features.append(col_name)
            if col_name == specification.class_column:
                continue
            all_data = dataset[col_name].values
            training_data = all_data[training_mask]
            self.pre_processing.fit_categorical(col_name, training_data)
            new_values = self.pre_processing.transform_categorical(col_name, all_data, self.input_encoding.required_input_format)

            if self.input_encoding.required_input_format == CategoricalFormat.OneHot:
                if isinstance(new_values, pd.DataFrame):
                    for c in new_values.columns:
                        new_df[c] = new_values[c]
                else:
                    for z in range(new_values.shape[1]):
                        new_df[f"{col_name}_{z}"] = new_values[:, z]
            else:
                new_df[col_name] = new_values

        new_df = pd.DataFrame(new_df)
        model_input_spec = ModelInputSpecification(new_features, len(numerical_columns), levels_per_categorical_feature, self.input_encoding.required_input_format)

        if cache_file_path is not None:
            print(f"Writing to cache file path: {cache_file_path}...")
            save_feather_plus_metadata(cache_file_path, new_df, model_input_spec)

        if self.retain_inmem_cache:
            if self.inmem_cache is None:
                self.inmem_cache = {}
            self.inmem_cache[cache_key] = (new_df, model_input_spec)

        return new_df, model_input_spec

    def load_dataset(self, dataset_name: str,
                     dataset: Union[pd.DataFrame, str],
                     specification,  # Assume DatasetSpecification type
                     cache_path: Optional[str] = None,
                     n_rows: int = 0,
                     evaluation_dataset_sampling=None,  # Assume EvaluationDatasetSampling type
                     evaluation_percent: float = 0.2,
                     numerical_filter: int = 1_000_000_000) -> pd.DataFrame:
        """
        加载数据集并为训练做好准备
        """
        if cache_path is None:
            cache_path = "cache"

        if not os.path.exists(cache_path):
            warnings.warn(f"Could not find cache folder: {cache_path}, attempting to create")
            os.mkdir(cache_path)

        self.dataset_specification = specification
        df, model_input_spec = self._load_preprocessed_dataset(dataset_name, dataset, specification, cache_path, n_rows, evaluation_dataset_sampling, evaluation_percent, numerical_filter)

        training_mask = df["__training"].values
        del df["__training"]

        y = df["__y"].values
        del df["__y"]

        self.X = df
        self.y = y
        self.training_mask = training_mask
        self.model_input_spec = model_input_spec

        return df