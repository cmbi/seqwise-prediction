#!/usr/bin/env python

from argparse import ArgumentParser
import os
import logging
from math import log

import numpy
import pandas
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.optim import Adam

from seqwise.dataset import SequenceDataset
from seqwise.model import (TransformerEncoderModel,
                           ReswiseModel,
                           FlatteningModel,
                           RelativeAbsolutePositionEncodingModel,
                           RelativePositionEncodingModel,
                           OuterSumModel)


arg_parser = ArgumentParser(description="train a model and output the results")
arg_parser.add_argument("model_type", help="must be outersum/relative/relabs/transformer/reswise/flattening")
arg_parser.add_argument("train_file", help="HDF5 file with training data")
arg_parser.add_argument("valid_file", help="HDF5 file with validation data")
arg_parser.add_argument("test_file", help="HDF5 file with test data")
arg_parser.add_argument("results_file", help="CSV file where results will be stored")
arg_parser.add_argument("--batch_size", "-b", type=int, default=64)
arg_parser.add_argument("--epoch-count", "-e", type=int, default=100)
arg_parser.add_argument("--classification", "-c", action="store_const", const=True, default=False)
arg_parser.add_argument("--blosum", help="use blosum62 encoding instead of one-hot encoding", action="store_const", const=True, default=False)


_log = logging.getLogger(__name__)


def get_model(model_type: str, classification: bool):

    if model_type == "transformer":
        return TransformerEncoderModel(classification)

    elif model_type == "outersum":
        return OuterSumModel(classification)

    elif model_type == "relabs":
        return RelativeAbsolutePositionEncodingModel(classification)

    elif model_type == "relative":
        return RelativePositionEncodingModel(classification)

    elif model_type == "reswise":
        return ReswiseModel(classification)

    elif model_type == "flattening":
        return FlatteningModel(classification)

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


def epoch(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          data_loader: DataLoader,
          epoch_index: int,
          metrics_path: str,
          phase_name: str,
          classification: bool):

    if classification:
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_func = torch.nn.MSELoss(reduction='mean')
        affinity_treshold = 1.0 - log(500) / log(50000)

    epoch_loss = 0.0
    epoch_true = []
    epoch_true_cls = []
    epoch_pred = []
    epoch_pred_cls = []
    for input_, true in data_loader:

        optimizer.zero_grad()

        output = model(input_)
        batch_loss = loss_func(output, true)

        batch_loss.backward()
        optimizer.step()

        batch_size = true.shape[0]
        epoch_loss += batch_loss.item() * batch_size

        if classification:
            epoch_pred += output.tolist()
            epoch_true += true.tolist()
        else:
            epoch_pred += output[..., 0].tolist()
            epoch_true += true[..., 0].tolist()

    if classification:
        auc = roc_auc_score(epoch_true, epoch_pred)
        mcc = matthews_corrcoef(epoch_true, torch.argmax(epoch_pred, dim=-1))
    else:
        pcc = pearsonr(epoch_true, epoch_pred).statistic
        store_metrics(metrics_path, phase_name, epoch_index, "pearson correlation", pcc)

        true_cls = numpy.array(epoch_true) > affinity_treshold
        pred_cls = numpy.array(epoch_pred) > affinity_treshold
        auc = roc_auc_score(true_cls, epoch_pred)
        mcc = matthews_corrcoef(true_cls, pred_cls)

    epoch_loss /= len(epoch_true)

    store_metrics(metrics_path, phase_name, epoch_index, "pearson correlation", pcc)
    store_metrics(metrics_path, phase_name, epoch_index, "ROC AUC", auc)
    store_metrics(metrics_path, phase_name, epoch_index, "matthews correlation", mcc)
    store_metrics(metrics_path, phase_name, epoch_index, "loss", epoch_loss)


def valid(model: torch.nn.Module,
          data_loader: DataLoader,
          epoch_index: int,
          metrics_path: str,
          phase_name: str,
          classification: bool):

    if classification:
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_func = torch.nn.MSELoss(reduction='mean')
        affinity_treshold = 1.0 - log(500) / log(50000)

    valid_loss = 0.0
    valid_true = []
    valid_true_cls = []
    valid_pred = []
    valid_pred_cls = []
    with torch.no_grad():
        for input_, true in data_loader:
            output = model(input_)
            batch_loss = loss_func(output, true)

            batch_size = true.shape[0]
            valid_loss += batch_loss.item() * batch_size

            if classification:
                valid_pred += output.tolist()
                valid_true += true.tolist()
            else:
                valid_pred += output[..., 0].tolist()
                valid_true += true[..., 0].tolist()

    if classification:
        auc = roc_auc_score(valid_true, valid_pred)
        mcc = matthews_corrcoef(valid_true, torch.argmax(valid_pred, dim=-1))
    else:
        pcc = pearsonr(valid_true, valid_pred).statistic
        store_metrics(metrics_path, phase_name, epoch_index, "pearson correlation", pcc)

        true_cls = numpy.array(valid_true) > affinity_treshold
        pred_cls = numpy.array(valid_pred) > affinity_treshold
        auc = roc_auc_score(true_cls, valid_pred)
        mcc = matthews_corrcoef(true_cls, pred_cls)

    valid_loss /= len(valid_true)

    store_metrics(metrics_path, phase_name, epoch_index, "pearson correlation", pcc)
    store_metrics(metrics_path, phase_name, epoch_index, "ROC AUC", auc)
    store_metrics(metrics_path, phase_name, epoch_index, "mathews correlation", mcc)
    store_metrics(metrics_path, phase_name, epoch_index, "loss", valid_loss)


def train(model: torch.nn.Module,
          metrics_path: str,
          train_data_loader: DataLoader,
          valid_data_loader: DataLoader,
          test_data_loader: DataLoader,
          epoch_count: int,
          classification: bool):

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch_index in range(epoch_count):
        epoch(model, optimizer, train_data_loader, epoch_index, metrics_path, "train", classification)
        valid(model, valid_data_loader, epoch_index, metrics_path, "valid", classification)
        valid(model, test_data_loader, epoch_index, metrics_path, "test", classification)



if __name__ == "__main__":

    args = arg_parser.parse_args()

    train_dataset = SequenceDataset(args.train_file, args.classification)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = SequenceDataset(args.valid_file, args.classification)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    test_dataset = SequenceDataset(args.test_file, args.classification)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = get_model(args.model_type, args.classification)

    train(model, args.results_file, train_data_loader, valid_data_loader, test_data_loader, args.epoch_count, args.classification)
