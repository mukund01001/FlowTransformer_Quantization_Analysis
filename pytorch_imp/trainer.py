import torch
import pandas as pd
import argparse
import pdb
import os
import sys

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.flow_transformer import Transformer
from src.enumerations import EvaluationDatasetSampling
from src.preprocess import StandardPreProcessing
from src.dataloader import DatasetLoader
from src.dataset_specification import NamedDatasetSpecifications
from src.input_encodings import *

class ModelArgments:
    def __init__(self, input_dim=0):
        super().__init__()
        self.num_layers = 12
        self.embed_size = 768
        self.heads = 12
        self.dropout = 0.02
        self.input_dim = input_dim
        self.max_length = 512
        self.num_classes = 2
        self.forward_expansion = 4

def check_dim(train_loader: DataLoader) -> int:
    for x, y in train_loader:
        return x.shape[-1]

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CSE_CIC_IDS")
    parser.add_argument("--data_path", type=str, default="../datasets.csv")
    parser.add_argument("--dataset_specification", type=NamedDatasetSpecifications, default=NamedDatasetSpecifications.unified_flow_format)
    parser.add_argument("--eval_percent", type=float, default=0.01)
    parser.add_argument("--eval_method", type=EvaluationDatasetSampling, default=EvaluationDatasetSampling.LastRows)
    parser.add_argument("--encodings", type=BaseInputEncoding, default=NoInputEncoder())
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_categorical_levels", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    return args

def trainer(model, train_loader, eval_loader, criterion, optimizer, device, epochs=10, grad_accumulation_steps=4):
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(train_loader):
            X, y = data
            # 如果X和y都为0时，跳过这一批
            if torch.all(X == 0) and torch.all(y == 0):
                continue

            X = X.squeeze(0).to(device)
            y = y.squeeze(0).to(device)
            
            optimizer.zero_grad()  # 清除上一次的梯度
            
            output = model(X)
            loss = criterion(output, y.long())
            loss.backward()  # 反向传播

            # 如果累积的步数达到了预定值，执行一次梯度更新
            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清除梯度，为下一次累计准备
            
            total_loss += loss.item()
            
            # 每100步输出一次当前的loss
            if i % 10000 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")

        # 每个epoch结束后打印一下累计的损失
        print(f"Epoch {epoch} - Average Loss: {total_loss / len(train_loader)}")
    
        # evaluation
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                X, y = data
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                loss = criterion(output, y)
                if i % 10 == 0:
                    print(f"Evaluation Loss: {loss.item()}")

def main(args):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load dataset
    dataset = pd.read_csv(args.data_path)
    pre_processing = StandardPreProcessing(n_categorical_levels=args.n_categorical_levels)
    dataset_loader = DatasetLoader(dataset=dataset, 
                                   specification=args.dataset_specification, 
                                   pre_processing=pre_processing, 
                                   input_encoding=args.encodings, 
                                   evaluation_dataset_sampling=args.eval_method, 
                                   batch_size=args.batch_size, 
                                   num_workers=args.num_workers, 
                                   evaluation_percent=args.eval_percent, 
                                   window_size=args.window_size,
                                   )
    train_loader, eval_loader = dataset_loader.load_data()
    
    # load model
    input_dim = check_dim(train_loader)
    model_args = ModelArgments(input_dim=input_dim)
    model = Transformer(model_args)

    # loss function
    criterion = CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # training
    trainer(model, train_loader, eval_loader, criterion, optimizer, device, epochs=args.epochs)


if __name__ == "__main__":
    args = setup()
    main(args)
    

