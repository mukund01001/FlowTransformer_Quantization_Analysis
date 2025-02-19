import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import os
import time
import warnings

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from framework.base_classification_head import BaseClassificationHead
from framework.base_input_encoding import BaseInputEncoding
from framework.base_preprocessing import BasePreProcessing
from framework.dataset_specification import DatasetSpecification
from framework.enumerations import EvaluationDatasetSampling, CategoricalFormat
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from framework.model_input_specification import ModelInputSpecification
from framework.utilities import get_identifier, load_feather_plus_metadata, save_feather_plus_metadata



class FlowTransformer(nn.Module):
    def __init__(self, pre_processing: BasePreProcessing,
                 input_encoding: BaseInputEncoding,
                 sequential_model: FunctionalComponent,
                 classification_head: BaseClassificationHead,
                 params: FlowTransformerParameters):
        super(FlowTransformer, self).__init__()
        
        self.classification_head = classification_head
        self.sequential_model = sequential_model
        self.input_encoding = input_encoding
        self.pre_processing = pre_processing
        self.parameters = params

        self.dataset_specification = None
        self.X = None
        self.y = None
        self.training_mask = None
        self.model_input_spec = None

        # Define your model layers using transformers and PyTorch
        self.transformer_model = AutoModel.from_pretrained("bert-base-uncased")  # Example, replace with appropriate model
        self.classification_layers = nn.Sequential(
            nn.Linear(self.transformer_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(params.mlp_dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        # The forward pass of the transformer model
        transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
        output = transformer_output.last_hidden_state  # Or use pooler_output based on your need
        output = output.mean(dim=1)  # Pooling (simple mean)
        output = self.classification_layers(output)
        return output

    def _load_preprocessed_dataset(self, dataset_name:str,
                     dataset:Union[pd.DataFrame, str],
                     specification:DatasetSpecification,
                     cache_folder:Optional[str]=None,
                     n_rows:int=0,
                     evaluation_dataset_sampling:EvaluationDatasetSampling=EvaluationDatasetSampling.LastRows,
                     evaluation_percent:float=0.2,
                     numerical_filter=1_000_000_000) -> Tuple[pd.DataFrame, ModelInputSpecification]:

        cache_file_path = None

        if dataset_name is None:
            raise Exception(f"Dataset name must be specified so FlowTransformer can optimise operations between subsequent calls!")

        pp_key = get_identifier(
            {
                "__preprocessing_name": self.pre_processing.name,
                **self.pre_processing.parameters
            }
        )

        local_key = get_identifier({
            "evaluation_percent": evaluation_percent,
            "numerical_filter": numerical_filter,
            "categorical_method": str(self.input_encoding.required_input_format),
            "n_rows": n_rows,
        })

        cache_key = f"{dataset_name}_{n_rows}_{pp_key}_{local_key}"

        if FlowTransformer.retain_inmem_cache:
            if FlowTransformer.inmem_cache is not None and cache_key in FlowTransformer.inmem_cache:
                print(f"Using in-memory cached version of this pre-processed dataset. To turn off this functionality set FlowTransformer.retain_inmem_cache = False")
                return FlowTransformer.inmem_cache[cache_key]

        if cache_folder is not None:
            cache_file_name = f"{cache_key}.feather"
            cache_file_path = os.path.join(cache_folder, cache_file_name)

            print(f"Using cache file path: {cache_file_path}")

            if os.path.exists(cache_file_path):
                print(f"Reading directly from cache {cache_file_path}...")
                model_input_spec: ModelInputSpecification
                dataset, model_input_spec = load_feather_plus_metadata(cache_file_path)
                return dataset, model_input_spec

        if isinstance(dataset, str):
            print(f"Attempting to read dataset from path {dataset}...")
            if dataset.lower().endswith(".feather"):
                # read as a feather file
                dataset = pd.read_feather(dataset, columns=specification.include_fields+[specification.class_column])
            elif dataset.lower().endswith(".csv"):
                dataset = pd.read_csv(dataset, nrows=n_rows if n_rows > 0 else None)
            else:
                raise Exception("Unrecognised dataset filetype!")
        elif not isinstance(dataset, pd.DataFrame):
            raise Exception("Unrecognised dataset input type, should be a path to a CSV or feather file, or a pandas dataframe!")

        assert isinstance(dataset, pd.DataFrame)

        if 0 < n_rows < len(dataset):
            dataset = dataset.iloc[:n_rows]

        training_mask = np.ones(len(dataset),  dtype=bool)
        eval_n = int(len(dataset) * evaluation_percent)

        if evaluation_dataset_sampling == EvaluationDatasetSampling.FilterColumn:
            if dataset.columns[-1] != specification.test_column:
                raise Exception(f"Ensure that the 'test' ({specification.test_column}) column is the last column of the dataset being loaded, and that the name of this column is provided as part of the dataset specification")

        if evaluation_dataset_sampling != EvaluationDatasetSampling.LastRows:
            warnings.warn("Using EvaluationDatasetSampling options other than LastRows might leak some information during training, if for example the context window leading up to a particular flow contains an evaluation flow, and this flow has out of range values (out of range to when pre-processing was applied on the training flows), then the model might potentially learn to handle these. In any case, no class leakage is present.")

        if evaluation_dataset_sampling == EvaluationDatasetSampling.LastRows:
            training_mask[-eval_n:] = False
        elif evaluation_dataset_sampling == EvaluationDatasetSampling.RandomRows:
            index = np.arange(self.parameters.window_size, len(dataset))
            sample = self.rs.choice(index, eval_n, replace=False)
            training_mask[sample] = False
        elif evaluation_dataset_sampling == EvaluationDatasetSampling.FilterColumn:
            # must be the last column of the dataset
            training_column = dataset.columns[-1]
            print(f"Using the last column {training_column} as the training mask column")

            v, c = np.unique(dataset[training_column].values,  return_counts=True)
            min_index = np.argmin(c)
            min_v = v[min_index]

            warnings.warn(f"Autodetected class {min_v} of {training_column} to represent the evaluation class!")

            eval_indices = np.argwhere(dataset[training_column].values == min_v).reshape(-1)
            eval_indices = eval_indices[(eval_indices > self.parameters.window_size)]

            training_mask[eval_indices] = False
            del dataset[training_column]

        numerical_columns = set(specification.include_fields).difference(specification.categorical_fields)
        categorical_columns = specification.categorical_fields

        print(f"Set y to = {specification.class_column}")
        new_df = {"__training": training_mask, "__y": dataset[specification.class_column].values}
        new_features = []

        print("Converting numerical columns to floats, and removing out of range values...")
        for col_name in numerical_columns:
            assert col_name in dataset.columns
            new_features.append(col_name)

            col_values = dataset[col_name].values
            col_values[~np.isfinite(col_values)] = 0
            col_values[col_values < -numerical_filter] = 0
            col_values[col_values > numerical_filter] = 0
            col_values = col_values.astype("float32")

            if not np.all(np.isfinite(col_values)):
                raise Exception("Flow format data had non finite values after float transformation!")

            new_df[col_name] = col_values

        print(f"Applying pre-processing to numerical values")
        for i, col_name in enumerate(numerical_columns):
            print(f"[Numerical {i+1:,} / {len(numerical_columns)}] Processing numerical column {col_name}...")
            all_data = new_df[col_name]
            training_data = all_data[training_mask]

            self.pre_processing.fit_numerical(col_name, training_data)
            new_df[col_name] = self.pre_processing.transform_numerical(col_name, all_data)

        print(f"Applying pre-processing to categorical values")
        levels_per_categorical_feature = []
        for i, col_name in enumerate(categorical_columns):
            new_features.append(col_name)
            if col_name == specification.class_column:
                continue
            print(f"[Categorical {i+1:,} / {len(categorical_columns)}] Processing categorical column {col_name}...")

            all_data = dataset[col_name].values
            training_data = all_data[training_mask]

            self.pre_processing.fit_categorical(col_name, training_data)
            new_values = self.pre_processing.transform_categorical(col_name, all_data, self.input_encoding.required_input_format)

            if self.input_encoding.required_input_format == CategoricalFormat.OneHot:
                # multiple columns of one hot values
                if isinstance(new_values, pd.DataFrame):
                    levels_per_categorical_feature.append(len(new_values.columns))
                    for c in new_values.columns:
                        new_df[c] = new_values[c]
                else:
                    n_one_hot_levels = new_values.shape[1]
                    levels_per_categorical_feature.append(n_one_hot_levels)
                    for z in range(n_one_hot_levels):
                        new_df[f"{col_name}_{z}"] = new_values[:, z]
            else:
                # single column of integers
                levels_per_categorical_feature.append(len(np.unique(new_values)))
                new_df[col_name] = new_values

        print(f"Generating pre-processed dataframe...")
        new_df = pd.DataFrame(new_df)
        model_input_spec = ModelInputSpecification(new_features, len(numerical_columns), levels_per_categorical_feature, self.input_encoding.required_input_format)

        print(f"Input data frame had shape ({len(dataset)},{len(dataset.columns)}), output data frame has shape ({len(new_df)},{len(new_df.columns)}) after pre-processing...")

        if cache_file_path is not None:
            print(f"Writing to cache file path: {cache_file_path}...")
            save_feather_plus_metadata(cache_file_path, new_df, model_input_spec)

        if FlowTransformer.retain_inmem_cache:
            if FlowTransformer.inmem_cache is None:
                FlowTransformer.inmem_cache = {}

            FlowTransformer.inmem_cache.clear()
            FlowTransformer.inmem_cache[cache_key] = (new_df, model_input_spec)

        return new_df, model_input_spec

    def load_dataset(self, dataset_name:str,
                     dataset:Union[pd.DataFrame, str],
                     specification:DatasetSpecification,
                     cache_path:Optional[str]=None,
                     n_rows:int=0,
                     evaluation_dataset_sampling:EvaluationDatasetSampling=EvaluationDatasetSampling.LastRows,
                     evaluation_percent:float=0.2,
                     numerical_filter=1_000_000_000) -> pd.DataFrame:
        """
        Load a dataset and prepare it for training

        :param dataset: The path to a CSV dataset to load from, or a dataframe
        :param cache_path: Where to store a cached version of this file
        :param n_rows: The number of rows to ingest from the dataset, or 0 to ingest all
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

    def evaluate(self, model: nn.Module, batch_size, early_stopping_patience:int=5, epochs:int=100, steps_per_epoch:int=128):
        n_malicious_per_batch = int(0.5 * batch_size)
        n_legit_per_batch = batch_size - n_malicious_per_batch

        overall_y_preserve = np.zeros(dtype="float32", shape=(n_malicious_per_batch + n_legit_per_batch,))
        overall_y_preserve[:n_malicious_per_batch] = 1.

        selectable_mask = np.zeros(len(self.X), dtype=bool)
        selectable_mask[self.parameters.window_size:-self.parameters.window_size] = True
        train_mask = self.training_mask

        y_mask = ~(self.y.astype('str') == str(self.dataset_specification.benign_label))

        indices_train = np.argwhere(train_mask).reshape(-1)
        malicious_indices_train = np.argwhere(train_mask & y_mask & selectable_mask).reshape(-1)
        legit_indices_train = np.argwhere(train_mask & ~y_mask & selectable_mask).reshape(-1)

        indices_test: np.ndarray = np.argwhere(~train_mask).reshape(-1)

        optimizer = optim.Adam(model.parameters(), lr=1e-5)  # You can adjust the learning rate
        criterion = nn.BCELoss()

        def get_windows_for_indices(indices: np.ndarray, ordered):
            X: List[pd.DataFrame] = []

            if ordered:
                for i1 in indices:
                    X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
            else:
                context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
                                                        replace=False).reshape(-1)
                context_indices_batch[:, -1] = indices

                for index in context_indices_batch:
                    X.append(self.X.iloc[index])

            return X

        def samplewise_to_featurewise(X):
            sequence_length = len(X[0])

            combined_df = pd.concat(X)
            featurewise_X = []

            feature_columns_map = {}

            if len(feature_columns_map) == 0:
                for feature in self.model_input_spec.feature_names:
                    if feature in self.model_input_spec.numeric_feature_names or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
                        feature_columns_map[feature] = feature
                    else:
                        feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

            for feature in self.model_input_spec.feature_names:
                feature_columns = feature_columns_map[feature]
                combined_values = combined_df[feature_columns].values

                combined_values = np.array([combined_values[i:i + sequence_length] for i in range(0, len(combined_values), sequence_length)])
                featurewise_X.append(combined_values)

            return featurewise_X

        print(f"Building eval dataset...")
        eval_X = get_windows_for_indices(indices_test, True)
        print(f"Splitting dataset to featurewise...")
        eval_featurewise_X = samplewise_to_featurewise(eval_X)
        eval_y = y_mask[indices_test]
        eval_P = eval_y
        n_eval_P = np.count_nonzero(eval_P)
        eval_N = ~eval_y
        n_eval_N = np.count_nonzero(eval_N)
        print(f"Evaluation dataset is built!")

        print(f"Positive samples in eval set: {n_eval_P}")
        print(f"Negative samples in eval set: {n_eval_N}")

        epoch_results = []

        def run_evaluation(epoch):
            model.eval()  # Switch to evaluation mode
            with torch.no_grad():  # Disable gradient computation for evaluation
                pred_y = model(torch.tensor(eval_featurewise_X, dtype=torch.float32))
                pred_y = pred_y.squeeze().numpy() > 0.5  # Assuming binary classification

                pred_P = pred_y
                n_pred_P = np.count_nonzero(pred_P)

                pred_N = ~pred_y
                n_pred_N = np.count_nonzero(pred_N)

                TP = np.count_nonzero(pred_P & eval_P)
                FP = np.count_nonzero(pred_P & ~eval_P)
                TN = np.count_nonzero(pred_N & eval_N)
                FN = np.count_nonzero(pred_N & ~eval_N)

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                balanced_accuracy = (sensitivity + specificity) / 2

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0

                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print(f"Epoch {epoch} yielded predictions: {pred_y.shape}, overall balanced accuracy: {balanced_accuracy * 100:.2f}%, TP = {TP:,} / {n_eval_P:,}, TN = {TN:,} / {n_eval_N:,}")

                epoch_results.append({
                    "epoch": epoch,
                    "P": n_eval_P,
                    "N": n_eval_N,
                    "pred_P": n_pred_P,
                    "pred_N": n_pred_N,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                    "bal_acc": balanced_accuracy,
                    "f1": f1_score
                })

        class BatchYielder():
            def __init__(self, ordered, random, rs):
                self.ordered = ordered
                self.random = random
                self.cursor_malicious = 0
                self.cursor_legit = 0
                self.rs = rs

            def get_batch(self):
                malicious_indices_batch = self.rs.choice(malicious_indices_train, size=n_malicious_per_batch,
                                                        replace=False) \
                    if self.random else \
                    malicious_indices_train[self.cursor_malicious:self.cursor_malicious + n_malicious_per_batch]

                legitimate_indices_batch = self.rs.choice(legit_indices_train, size=n_legit_per_batch, replace=False) \
                    if self.random else \
                    legit_indices_train[self.cursor_legit:self.cursor_legit + n_legit_per_batch]

                indices = np.concatenate([malicious_indices_batch, legitimate_indices_batch])

                self.cursor_malicious = self.cursor_malicious + n_malicious_per_batch
                self.cursor_malicious = self.cursor_malicious % (len(malicious_indices_train) - n_malicious_per_batch)

                self.cursor_legit = self.cursor_legit + n_legit_per_batch
                self.cursor_legit = self.cursor_legit % (len(legit_indices_train) - n_legit_per_batch)

                X = get_windows_for_indices(indices, self.ordered)
                featurewise_X = samplewise_to_featurewise(X)

                return featurewise_X, overall_y_preserve

        batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows,
                                    not self.parameters._train_draw_sequential_windows, self.rs)

        min_loss = 100
        iters_since_loss_decrease = 0
        train_results = []
        final_epoch = 0
        last_print = time.time()
        elapsed_time = 0

        for epoch in range(epochs):
            final_epoch = epoch
            has_reduced_loss = False
            for step in range(steps_per_epoch):
                batch_X, batch_y = batch_yielder.get_batch()

                t0 = time.time()
                optimizer.zero_grad()  # Zero the gradients
                batch_results = model(batch_X)  # Forward pass
                loss = criterion(batch_results, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimization step
                t1 = time.time()

                elapsed_time += (t1 - t0)
                train_results.append([loss.item(), elapsed_time, epoch])

                if loss.item() < min_loss:
                    has_reduced_loss = True
                    min_loss = loss.item()

            if has_reduced_loss:
                iters_since_loss_decrease = 0
            else:
                iters_since_loss_decrease += 1

            do_early_stop = early_stopping_patience > 0 and iters_since_loss_decrease > early_stopping_patience
            is_last_epoch = epoch == epochs - 1
            run_eval = epoch in [6] or is_last_epoch or do_early_stop

            if run_eval:
                run_evaluation(epoch)

            if do_early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        eval_results = pd.DataFrame(epoch_results)

        return train_results, eval_results, final_epoch

    # def time(self, model: nn.Module, batch_size, n_steps=128, n_repeats=4):

    #     n_malicious_per_batch = int(0.5 * batch_size)
    #     n_legit_per_batch = batch_size - n_malicious_per_batch

    #     overall_y_preserve = np.zeros(dtype="float32", shape=(n_malicious_per_batch + n_legit_per_batch,))
    #     overall_y_preserve[:n_malicious_per_batch] = 1.

    #     selectable_mask = np.zeros(len(self.X), dtype=bool)
    #     selectable_mask[self.parameters.window_size:-self.parameters.window_size] = True
    #     train_mask = self.training_mask

    #     y_mask = ~(self.y.astype('str') == str(self.dataset_specification.benign_label))

    #     indices_train = np.argwhere(train_mask).reshape(-1)
    #     malicious_indices_train = np.argwhere(train_mask & y_mask & selectable_mask).reshape(-1)
    #     legit_indices_train = np.argwhere(train_mask & ~y_mask & selectable_mask).reshape(-1)

    #     indices_test: np.ndarray = np.argwhere(~train_mask).reshape(-1)

    #     def get_windows_for_indices(indices: np.ndarray, ordered):
    #         X: List[pd.DataFrame] = []

    #         if ordered:
    #             for i1 in indices:
    #                 X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
    #         else:
    #             context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
    #                                                     replace=False).reshape(-1)
    #             context_indices_batch[:, -1] = indices

    #             for index in context_indices_batch:
    #                 X.append(self.X.iloc[index])

    #         return X

    #     def samplewise_to_featurewise(X):
    #         sequence_length = len(X[0])

    #         combined_df = pd.concat(X)
    #         featurewise_X = []

    #         feature_columns_map = {}

    #         if len(feature_columns_map) == 0:
    #             for feature in self.model_input_spec.feature_names:
    #                 if feature in self.model_input_spec.numeric_feature_names or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
    #                     feature_columns_map[feature] = feature
    #                 else:
    #                     feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

    #         for feature in self.model_input_spec.feature_names:
    #             feature_columns = feature_columns_map[feature]
    #             combined_values = combined_df[feature_columns].values

    #             combined_values = np.array([combined_values[i:i + sequence_length] for i in range(0, len(combined_values), sequence_length)])
    #             featurewise_X.append(combined_values)

    #         return featurewise_X

    #     epoch_results = []
    #     class BatchYielder():
    #         def __init__(self, ordered, random, rs):
    #             self.ordered = ordered
    #             self.random = random
    #             self.cursor_malicious = 0
    #             self.cursor_legit = 0
    #             self.rs = rs

    #         def get_batch(self):
    #             malicious_indices_batch = self.rs.choice(malicious_indices_train, size=n_malicious_per_batch,
    #                                                      replace=False) \
    #                 if self.random else \
    #                 malicious_indices_train[self.cursor_malicious:self.cursor_malicious + n_malicious_per_batch]

    #             legitimate_indices_batch = self.rs.choice(legit_indices_train, size=n_legit_per_batch, replace=False) \
    #                 if self.random else \
    #                 legit_indices_train[self.cursor_legit:self.cursor_legit + n_legit_per_batch]

    #             indices = np.concatenate([malicious_indices_batch, legitimate_indices_batch])

    #             self.cursor_malicious = self.cursor_malicious + n_malicious_per_batch
    #             self.cursor_malicious = self.cursor_malicious % (len(malicious_indices_train) - n_malicious_per_batch)

    #             self.cursor_legit = self.cursor_legit + n_legit_per_batch
    #             self.cursor_legit = self.cursor_legit % (len(legit_indices_train) - n_legit_per_batch)

    #             X = get_windows_for_indices(indices, self.ordered)
    #             # each x in X contains a dataframe, with window_size rows and all the features of the flows. There are batch_size of these.

    #             # we have a dataframe containing batch_size x (window_size, features)
    #             # we actually want a result of features x (batch_size, sequence_length, feature_dimension)
    #             featurewise_X = samplewise_to_featurewise(X)

    #             return featurewise_X, overall_y_preserve

    #     batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows, not self.parameters._train_draw_sequential_windows, self.rs)
    #     min_loss = 100
    #     iters_since_loss_decrease = 0

    #     final_epoch = 0

    #     last_print = time.time()
    #     elapsed_time = 0

    #     batch_times = []


    #     for step in range(n_steps):
    #         batch_X, batch_y = batch_yielder.get_batch()

    #         local_batch_times = []
    #         for i in range(n_repeats):
    #             t0 = time.time()
    #             batch_results = model(batch_X)  # Forward pass
    #             t1 = time.time()
    #             local_batch_times.append(t1 - t0)

    #         batch_times.append(local_batch_times)

    #         if time.time() - last_print > 3:
    #             last_print = time.time()
    #             print(f"Step = {step}, running model evaluation... Average times = {np.mean(np.array(batch_times).reshape(-1))}")

    #     return batch_times