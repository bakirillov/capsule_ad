"""The script to reproduce the model training.
This file provides the code to train every model.
This file is to be ran as a standalone script.
"""

import os
import torch
import einops
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import os.path as op
from tqdm import tqdm
from skimage import io
from torch.optim import Adam
import pytorch_lightning as pl
from capsules.capsules import *
from skimage.transform import resize
from torchvision import models, transforms
from torch.nn.functional import tanh, sigmoid
from torch.utils.data import DataLoader, Dataset
from torch.nn import Conv2d, ReLU, ELU, LeakyReLU, Conv1d
from torch.nn import Sequential, Linear, Tanh, Sigmoid, MSELoss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import *
from data_module import *


# This defines transforms for the datasets
cifar_transforms = transforms.Compose(
    [
        lambda x: io.imread(x).astype(np.float32),
        lambda x: einops.rearrange(
            x, "h w c -> c h w"
        )/255.0
    ]
)
mnist_transforms = transforms.Compose(
    [
        lambda x: io.imread(x).astype(np.float32),
        lambda x: einops.rearrange(
            x, "h w -> 1 h w",
        )/255.0,
    ]
)
peng_transforms = transforms.Compose(
    [
        lambda x: np.stack([correct_order(onehot(a)) for a in x.split(",")]),
        lambda x: einops.rearrange(
            x, "c (h w) -> c h w", h = 4
        ).astype(np.float32)
    ]
)
ham_transforms = transforms.Compose(
    [
        lambda x: io.imread(x).astype(np.float32),
        lambda x: resize(x, (75, 100)),
        lambda x: einops.rearrange(
            x, "h w c -> c h w"
        )/255.0
    ]
)

#Transforms, batch sizes, models and paths are stored in dicts
#The distc are quieried by the command line arguments
TRANSFORMS = {
    "MNIST": mnist_transforms,
    "FMNIST": mnist_transforms,
    "KMNIST": mnist_transforms,
    "CIFAR10": cifar_transforms,
    "HAM10000": ham_transforms,
    "PENG": peng_transforms,
}

BATCH_SIZES = {
    "MNIST": 256,
    "FMNIST": 256,
    "KMNIST": 256,
    "CIFAR10": 64,
    "HAM10000": 64,
    "PENG": 256,
}

MODELS = {
    "S": {
        "NL": {
            "MNIST": lambda : NL((1,28,28), 500),
            "FMNIST": lambda : NL((1,28,28), 500),
            "KMNIST": lambda : NL((1,28,28), 500),
            "CIFAR10": lambda : NL((3,32,32), 500),
            "HAM10000": lambda : NL((3,75,100), 500),
            "PENG": lambda : NL((2,4,23), 500),
        },
        "ABC": {
            "MNIST": lambda : ABC(input_shape=(28*28), nll=False),
            "FMNIST": lambda : ABC(input_shape=(28*28), nll=False),
            "KMNIST": lambda : ABC(input_shape=(28*28), nll=False),
            "CIFAR10": lambda : ABC(input_shape=(3*32*32), nll=False),
            "HAM10000": lambda : ABC(input_shape=(3*75*100), nll=False),
            "PENG": lambda : ABC(input_shape=(2*4*23), nll=False),
        },
        "CN": {
            "MNIST": lambda : CapsNet4AD(1152, 1, 28*28),
            "FMNIST": lambda : CapsNet4AD(1152, 1, 28*28),
            "KMNIST": lambda : CapsNet4AD(1152, 1, 28*28),
            "CIFAR10": lambda : CapsNet4AD(2048, 3, 3*32*32),
            "HAM10000": lambda : CapsNet4HAM(40320, 3*100*75),
            "PENG": lambda : CapsNet4AD(352, 2, 2*4*23, kernel_size=2, primcaps_kernel_size=2),
        }
    },
    "U": {
        "DI": {
            "CN": {
                "MNIST": lambda : CapsNet4AD(1152, 1, 28*28, n_classes=9),
                "FMNIST": lambda : CapsNet4AD(1152, 1, 28*28, n_classes=9),
                "KMNIST": lambda : CapsNet4AD(1152, 1, 28*28, n_classes=9),
                "CIFAR10": lambda : CapsNet4AD(2048, 3, 3*32*32, n_classes=9),
                "HAM10000": lambda : CapsNet4HAM(40320, 3*100*75),
            }
        },
        "DO": {
            "CN": {
                "MNIST": lambda : CapsNet4AD(1152, 1, 28*28, n_classes=10),
                "FMNIST": lambda : CapsNet4AD(1152, 1, 28*28, n_classes=10),
                "KMNIST": lambda : CapsNet4AD(1152, 1, 28*28, n_classes=10),
                "CIFAR10": lambda : CapsNet4AD(2048, 3, 3*32*32, n_classes=10),
                "HAM10000": lambda : CapsNet4HAM(40320, 3*100*75),
            }
        }
    }
}


PATHS = {
    "MNIST": ("../data/MNIST_train.csv", "../data/MNIST_test.csv"),
    "FMNIST": ("../data/FMNIST_train.csv", "../data/FMNIST_test.csv"),
    "KMNIST": ("../data/KMNIST_train.csv", "../data/KMNIST_test.csv"),
    "CIFAR10": ("../data/CIFAR10_train.csv", "../data/CIFAR10_test.csv"),
    "HAM10000": ("../data/HAM10000_train.csv", "../data/HAM10000_test.csv"),
    "PENG": ("../data/PENG_train.csv", "../data/PENG_test.csv"),
}


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
    parser.add_argument(
        "-i", "--indices",
        dest="indices",
        action="store_true", 
        help="use indices"
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
        args.output, args.dataset+"_"+args.model+"_"+args.et+"_"+args.cl+"_"+args.proportion+"_"+args.seed+".pt"
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
    #Here, the dicts provided above are queried
    dm = DM(
        PATHS[args.dataset][0], 
        PATHS[args.dataset][1], 
        experiment_type, proportion, transforms, 
        batch_size=batch_size, use_indices=args.indices
    )
    if args.et[0] == "S":
        model = MODELS[args.et[0]][args.model][args.dataset]()
    else:
        model = MODELS[args.et[0]][args.et[1:]][args.model][args.dataset]()
    #NL models require a bit different approach in training
    #So for everything else, the training is done by Lightning trainer
    if args.model != "NL":
        es = EarlyStopping(monitor='val_loss', patience=3, verbose=True)
        trainer = pl.Trainer(
            gpus="1", max_epochs=int(args.epochs),
            default_root_dir=output_filename.replace(".pt", "_logs"),
            callbacks=[es], gradient_clip_val=0.1
        )
        trainer.fit(model, dm)
    else:
        #But for NLs, training code is as follows
        #Adapted from https://github.com/eugenet12/pytorch-rbm-autoencoder
        dm.prepare_data()
        dm.setup()
        dm.train_dataloader()
        prev_pn = 1
        improved = False
        j = 0
        for i in np.arange(int(args.epochs)):
            print(i, args.epochs, improved, j, prev_pn)
            this_epoch = []
            for a in tqdm(dm.train_loader):
                p,n = model.training_step(a, None)
                this_epoch.append(p.cpu().data.numpy())
            mnp = np.mean(this_epoch)
            if mnp < prev_pn:
                print("Loss improved from "+str(prev_pn)+" to "+str(mnp))
                prev_pn = mnp
                improved = True
            else:
                improved = False
            if not improved:
                j += 1
            else:
                j = 0
            if j >= 3:
                print("Loss haven't improved for 3 epoch. Stopping.")
                break
    torch.save(model.state_dict(), output_filename)
    print("State dictionary saved to "+output_filename)
