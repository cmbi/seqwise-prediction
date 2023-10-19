#!/usr/bin/env python

from argparse import ArgumentParser
import os
import logging

import pandas
import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from torch.optim import Adam

from seqwise.dataset import SequenceDataset
from seqwise.model import (TransformerEncoderModel,
                           ReswiseModel,
                           FlatteningModel,
                           RelativePositionEncodingModel,
                           OuterSumModel)


arg_parser = ArgumentParser(description="train a model and output the results")
arg_parser.add_argument("model_type", help="must be outersum/relative/transformer/reswise/flattening")
arg_parser.add_argument("train_file", help="HDF5 file with training data")
arg_parser.add_argument("valid_file", help="HDF5 file with validation data")
arg_parser.add_argument("test_file", help="HDF5 file with test data")
arg_parser.add_argument("results_file", help="CSV file where results will be stored")
arg_parser.add_argument("--batch_size", "-b", type=int, default=64)
arg_parser.add_argument("--epoch-count", "-e", type=int, default=100)


_log = logging.getLogger(__name__)


def get_model(model_type: str):

    if model_type == "transformer":
        return TransformerEncoderModel()

    elif model_type == "outersum":
        return OuterSumModel()

    elif model_type == "relative":
        return RelativePositionEncodingModel()

    elif model_type == "reswise":
        return ReswiseModel()

    elif model_type == "flattening":
        return FlatteningModel()

    else:
        raise ValueError(f"unknown model: {model_type}")

def store_metrics(path: str, phase_name: str, epoch_index: int, value_name: str, value: float):

    column_name = f"{phase_name} {value_name}"

    if os.path.isfile(path):
        table = pandas.read_csv(path)
    else:
        table = pandas.DataFrame({"epoch": [epoch_index], column_name: value})

    table.loc[epoch_index, "epoch"] = epoch_index
    table.loc[epoch_index, column_name] = value

    table.to_csv(path, index=False)

    _log.debug(f"store {column_name}, {value}")


loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

def epoch(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          data_loader: DataLoader,
          epoch_index: int,
          metrics_path: str,
          phase_name: str):

    epoch_loss = 0.0
    epoch_true = []
    epoch_pred = []
    for input_, true_cls in data_loader:

        optimizer.zero_grad()

        output = model(input_)
        batch_loss = loss_func(output, true_cls)
        pred_cls = torch.argmax(output, dim=1)

        batch_loss.backward()
        optimizer.step()

        batch_size = true_cls.shape[0]
        epoch_loss += batch_loss.item() * batch_size

        epoch_true += true_cls.tolist()
        epoch_pred += pred_cls.tolist()

    mcc = matthews_corrcoef(epoch_true, epoch_pred)
    epoch_loss /= len(epoch_true)

    store_metrics(metrics_path, phase_name, epoch_index, "matthews correlation", mcc)
    store_metrics(metrics_path, phase_name, epoch_index, "loss", epoch_loss)


def valid(model: torch.nn.Module,
          data_loader: DataLoader,
          epoch_index: int,
          metrics_path: str,
          phase_name: str):

    valid_loss = 0.0
    valid_true = []
    valid_pred = []
    with torch.no_grad():
        for input_, true_cls in data_loader:
            output = model(input_)
            batch_loss = loss_func(output, true_cls)
            pred_cls = torch.argmax(output, dim=1)

            batch_size = true_cls.shape[0]
            valid_loss += batch_loss.item() * batch_size

            valid_true += true_cls.tolist()
            valid_pred += pred_cls.tolist()

    mcc = matthews_corrcoef(valid_true, valid_pred)
    valid_loss /= len(valid_true)

    store_metrics(metrics_path, phase_name, epoch_index, "mathews correlation", mcc)
    store_metrics(metrics_path, phase_name, epoch_index, "loss", valid_loss)


def train(model: torch.nn.Module,
          metrics_path: str,
          train_data_loader: DataLoader,
          valid_data_loader: DataLoader,
          test_data_loader: DataLoader,
          epoch_count: int):

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch_index in range(epoch_count):
        epoch(model, optimizer, train_data_loader, epoch_index, metrics_path, "train")
        valid(model, valid_data_loader, epoch_index, metrics_path, "valid")
        valid(model, test_data_loader, epoch_index, metrics_path, "test")



if __name__ == "__main__":



    args = arg_parser.parse_args()

    train_dataset = SequenceDataset(args.train_file)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    valid_dataset = SequenceDataset(args.valid_file)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    test_dataset = SequenceDataset(args.test_file)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = get_model(args.model_type)

    train(model, args.results_file, train_data_loader, valid_data_loader, test_data_loader, args.epoch_count)
