from transformer import Transformer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

from typing import List, Tuple

from dataloader import CustomDataset
from model_input_specification import ModelInputSpecification
from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from preprocess import StandardPreProcessing
from dataloader import DatasetLoader
from input_encodings import *
import torch
import os
import math
import argparse

flow_file_path = "../../demonstration/b3427ed8ad063a09_MOHANAD_A4706/data/"
datasets = [
    ("CSE_CIC_IDS", os.path.join(flow_file_path, "NF-CSE-CIC-IDS2018-v2.csv"), NamedDatasetSpecifications.unified_flow_format, 0.01, EvaluationDatasetSampling.LastRows),
    ("NSL-KDD", os.path.join(flow_file_path, "NSL-KDD.csv"), NamedDatasetSpecifications.nsl_kdd, 0.05, EvaluationDatasetSampling.RandomRows),
    ("UNSW_NB15", os.path.join(flow_file_path, "NF-UNSW-NB15-v2.csv"), NamedDatasetSpecifications.unified_flow_format, 0.025, EvaluationDatasetSampling.LastRows)
]

encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    CategoricalFeatureEmbed(EmbedLayerType.Dense, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Lookup, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Projection, 16),
    RecordLevelEmbed(64, project=True)
]


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_specification", type=NamedDatasetSpecifications)
    parser.add_argument("--eval_percent", type=float)
    parser.add_argument("--eval_method", type=EvaluationDatasetSampling)



    args = parser.parse_args()
    args.dataset_name, args.data_path, args.dataset_specification, args.eval_percent, args.eval_method = datasets[0]

    return args


def trainer(args):

    pass

def main(args):
    # set parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the specific dataset
    dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
    


    


    

    



if __name__ == "__main__":
    args = setup()
    main(args)
    

