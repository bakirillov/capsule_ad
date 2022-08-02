"""The script to reproduce the computation of the metrics.
This file provides the code to compute metrics for every model.
This file is to be ran as a standalone script.
"""

import os
import json
import torch
import einops
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import os.path as op
from tqdm import tqdm
from skimage import io
from tensorly import norm
from torch.optim import Adam
import pytorch_lightning as pl
from capsules.capsules import *
from skimage.transform import resize
from torchvision import models, transforms
from torch.nn.functional import tanh, sigmoid
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import Conv2d, ReLU, ELU, LeakyReLU, Conv1d
from torch.nn import Sequential, Linear, Tanh, Sigmoid, MSELoss


from models import *
from data_module import *
from train_model import *


def G(x):
    """
    Computes Gini coefficient. Adapted from https://github.com/oliviaguest/gini
    x - a numpy.array
    Outputs a float value of G(x)
    """
    X = x.flatten()
    X = X - np.amin(X) if np.amin(X) < 0 else X
    X = np.sort(X + 10**-7)
    n = X.shape[0]
    i = np.arange(1, n+1)
    nom = np.sum((2*i-n-1)*X)
    den = n*np.sum(X)
    return(nom/den)

def Palma(x):
    """
    Computes Palma coefficient.
    x - a numpy.array
    Outputs a float value of P(x)
    """
    return(np.percentile(x, 90)/np.percentile(x, 40))

def apply_to_couplings(couplings, f):
    """
    Correctly map the function across the couplings.
    couplings - a numpy.array with the couplings.
    f - function to compute.
    Outputs a numpy.array of results.
    """
    out = []
    for a in np.arange(couplings.shape[1]):
        out.append(
            np.array([f(b[a].squeeze().sum(1)) for b in couplings])
        )
    return(np.stack(out).T)


def compute_metrics_CN(test_loader, model):
    """
    Compute the metrics over a test DataLoader.
    test_loader - a torch.utils.data.DataLoader.
    model - a Capsule Network.
    Outputs a pandas.DataFrame with the metrics.
    """
    test_metrics = {
        "plain": [],
        "plain_probs": [],
        "gini": [],
        "palma": [],
        "real": [],
        "n_pp": [],
        "n_re": [],
        "anomaly_score": [],
        "a_diff": [],
        "a_re": []
    }
    for i,batch in tqdm(enumerate(test_loader)):
        x,y = batch
        internal, reconstruction, lengths, max_caps_index, couplings = model(x.to(0))
        current_couplings = np.swapaxes(couplings, 0, 1)
        ginis = apply_to_couplings(current_couplings, G)
        palmas = apply_to_couplings(current_couplings, Palma)
        a_diff, a_re = anomaly_scores(
            lengths.cpu().data.numpy(),
            x.cpu().data.numpy().reshape(x.shape[0], -1),
            reconstruction.cpu().data.numpy()
        )
        anomalies = a_diff+a_re
        n_pp, n_re = normality_scores(
            lengths.cpu().data.numpy(),
            x.cpu().data.numpy().reshape(x.shape[0], -1),
            reconstruction.cpu().data.numpy()
        )
        test_metrics["plain"].extend(max_caps_index.cpu().data.numpy())
        test_metrics["plain_probs"].extend(lengths.cpu().data.numpy())
        test_metrics["real"].extend(y.cpu().data.numpy())
        test_metrics["gini"].extend(ginis)
        test_metrics["palma"].extend(palmas)
        test_metrics["anomaly_score"].extend(anomalies)
        test_metrics["a_diff"].extend(a_diff)
        test_metrics["a_re"].extend(a_re)
        test_metrics["n_pp"].extend(n_pp)
        test_metrics["n_re"].extend(n_re)
    return(pd.DataFrame(test_metrics))

def compute_metrics_ABC(test_loader, model):
    """
    Compute the metrics over a test DataLoader.
    test_loader - a torch.utils.data.DataLoader.
    model - an Autoencoding Binary Classifier.
    Outputs a pandas.DataFrame with the metrics.
    """
    test_metrics = {
        "real": [],
        "score": []
    }
    for i,batch in tqdm(enumerate(test_loader)):
        x,y = batch
        x_hat = model(x.to(0)).to("cpu").data.numpy()
        mse = np.mean((x.to("cpu").data.numpy()-x_hat)**2, (1,2,3))
        test_metrics["real"].extend(y.to("cpu").data.numpy().astype(np.int32))
        test_metrics["score"].extend(np.exp(-mse))
    return(pd.DataFrame(test_metrics))

def compute_metrics_NL(test_loader, model):
    """
    Compute the metrics over a test DataLoader.
    test_loader - a torch.utils.data.DataLoader.
    model - a Restricted Boltzmann Machine.
    Outputs a pandas.DataFrame with the metrics.
    """
    test_metrics = {
        "real": [],
        "score": []
    }
    for i,batch in tqdm(enumerate(test_loader)):
        x, y = batch
        x = x.reshape(x.shape[0], -1).squeeze()
        pvk, _, _ = model(x)
        pvk = pvk.to("cpu").data.numpy()
        mse = np.mean((x.to("cpu").data.numpy()-pvk)**2, 1)
        test_metrics["real"].extend(y.to("cpu").data.numpy().astype(np.int32))
        test_metrics["score"].extend(np.exp(-mse))
    return(pd.DataFrame(test_metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        dest="dataset",
        action="store", 
        help="set the dataset", 
        default="MNIST",
        choices=[
            "MNIST", "FMNIST", "KMNIST", "CIFAR10", "HAM10000", "PENG"
        ]
    )
    parser.add_argument(
        "-t", "--type",
        dest="et",
        action="store", 
        help="set the experiment type", 
        default=["SDI"], 
        choices=[
            "SDI", "SDO",
            "UDI", "UDO",
            "SA", "SB",
            "SC", "SD",
            "UA", "UB",
            "UC", "UD",
        ],
    )
    parser.add_argument(
        "-m", "--model",
        dest="model",
        action="store", 
        help="set the model to train", 
        default=["CN"], 
        choices=["CN", "ABC", "NL"],
    )
    parser.add_argument(
        "-l", "--label",
        dest="cl",
        action="store", 
        help="set the chosen label", 
        default="0",
    )
    parser.add_argument(
        "-e", "--epochs",
        dest="epochs",
        action="store", 
        help="set the number of epochs", 
        default="10"
    )
    parser.add_argument(
        "-p", "--proportion",
        dest="proportion",
        action="store", 
        help="set the proportion of anomalies", 
        default="None"
    )
    parser.add_argument(
        "-i", "--indices",
        dest="indices",
        action="store_true", 
        help="use indices"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store", 
        help="set the path of output directory"
    )
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        action="store", 
        help="set the random seed"
    )
    args = parser.parse_args()
    seed = int(args.seed)
    pl.utilities.seed.seed_everything(seed)
    if not op.exists(args.output):
        os.makedirs(args.output)
        print("Created "+args.output)
    if args.dataset != "HAM10000":
        if args.dataset != "PENG":
            experiment_type = args.et+"_"+args.cl
        else:
            experiment_type = "original_label"
    else:
        experiment_type = args.et[1:]
    output_filename = op.join(
        args.output, 
        args.dataset+"_"+args.model+"_"+args.et+"_"+args.cl+"_"+args.proportion+"_"+args.seed+".csv"
    )
    if args.proportion != "None":
        if float(args.proportion) >= 1:
            proportion = int(args.proportion)
        else:
            proportion = float(args.proportion)
    else:
        proportion = None
    batch_size = BATCH_SIZES[args.dataset]
    transforms = TRANSFORMS[args.dataset]
    print(args.dataset, args.model, args.et, proportion)
    dm = DM(
        PATHS[args.dataset][0], 
        PATHS[args.dataset][1], 
        experiment_type, proportion, transforms, 
        batch_size=16, use_indices=args.indices
    )
    if args.et[0] == "S":
        model = MODELS[args.et[0]][args.model][args.dataset]().to(0)
    else:
        model = MODELS[args.et[0]][args.et[1:]][args.model][args.dataset]().to(0)
    model.load_state_dict(
        torch.load(
            output_filename.replace(".csv", ".pt")
        )
    )
    dm.prepare_data()
    dm.setup()
    dm.train_dataloader()
    dm.test_dataloader()
    if args.model == "CN":
        train_metrics = compute_metrics_CN(dm.train_loader, model)
        test_metrics = compute_metrics_CN(dm.test_loader, model)
    elif args.model == "NL":
        train_metrics = compute_metrics_NL(dm.train_loader, model)
        test_metrics = compute_metrics_NL(dm.test_loader, model)
    else:
        train_metrics = compute_metrics_ABC(dm.train_loader, model)
        test_metrics = compute_metrics_ABC(dm.test_loader, model)
    train_metrics.to_csv(output_filename.replace(".csv", ".train.csv"))
    test_metrics.to_csv(output_filename.replace(".csv", ".test.csv"))
    print("Metrics saved to "+output_filename)
