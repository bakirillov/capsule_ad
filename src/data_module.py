"""The module with definitions of the datasets and data modules
This file provides the definitions to construct every dataset class.
This file is to be imported as a module and exports the following:
    DS - A torch Dataset for every dataset used in the study
    DM - A lightning DataModule for every dataset used in the study
"""


import numpy as np
import pandas as pd
import pickle as pkl
from skimage import io
import pytorch_lightning as pl
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


class DS(Dataset):
    """
    A torch Dataset for every dataset used in the study
    ---------------------------------------------------
    ...
    Attributes
    ----------
    transform : transforms.Compose of a function
        a transformation to apply for each example
    t_indices : [int]
        indices to get a subset of the data
    paths : [string]
        a list of strings that contain paths to files
        or gRNA-target pairs in case of Peng et al. dataset
    labels : [int]
        a list of class labels
            
    Methods
    -------
    __len__(self)   
        Number of pairs   
    __getitem__(self, ind)   
        Get a pair by its index   
    """
    
    def __init__(
        self, df, column, p=None, t_indices=None, transform=None, use_indices=True
    ):
        self.transform = transform
        self.t_indices = t_indices
        self.paths = df["path"].values
        self.labels = df[column].values.astype(np.float32)
        if use_indices:
            if p:
                self.paths = self.paths[t_indices[column+"_"+str(p)]]
                self.labels = self.labels[t_indices[column+"_"+str(p)]]
            else:
                self.paths = self.paths[t_indices[column]]
                self.labels = self.labels[t_indices[column]]
        
    def __len__(self):
        """Number of examples"""
        return(self.paths.shape[0])
    
    def __getitem__(self, ind):
        """
        Get an example by its index
        
        ind is the index of an example AFTER t_indices argument is applied
        
        Outputs a tuple of optionally transformed example and its label
        """
        path = self.paths[ind]
        label = self.labels[ind]
        if self.transform:
            path = self.transform(path)
        return(path, label)


class DM(pl.LightningDataModule):
    """
    A lightning DataModule for every dataset used in the study
    ----------------------------------------------------------
    ...
    Attributes
    ----------
    transform : transforms.Compose of a function
        a transformation to apply for each example - 
    p : float
        proportion of anomal labels
    train_path : string
        string with the training set path
    test_path : string
        string with the test set path
    batch_size : int
        number of examples in the batch
    use_indices : bool
        use the indices given in a separate file
            
    Methods
    -------
    prepare_data(self)
        Load the data from the hard drive
    setup(self, stage=None)
        Construct the final train and test sets
    train_dataloader(self):
        Construct the training set DataLoader
    test_dataloader(self):
        Construct the test set DataLoader
    val_dataloader(self):
        Construct the validation set DataLoader which is 
        the same as the test set because the study design 
        does not require a separate validation set
    """
    
    def __init__(
        self, train_path, test_path, column, p, transform, batch_size=64, use_indices=True
    ):
        super().__init__()
        self.p = p
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.column = column
        self.transform = transform
        self.use_indices = use_indices
        
    def prepare_data(self):
        """Load the data from the hard drive"""
        self.train_df = pd.read_csv(self.train_path, index_col=0)
        self.test_df = pd.read_csv(self.test_path, index_col=0)
        print("Train size", self.train_df.shape)
        print("Test size", self.test_df.shape)
        if self.use_indices:
            with open(self.train_path.replace(".csv", ".pkl"), "rb") as ih:
                self.t_indices = pkl.load(ih)
        else:
            self.t_indices = None
        
    def setup(self, stage=None):
        """Construct the final train and test sets"""
        ti = self.t_indices
        p = self.p
        self.train = DS(
            self.train_df, self.column, p, ti, self.transform, self.use_indices
        )
        print("Final train size", len(self.train))
        self.test = DS(
            self.test_df, self.column, None, None, transform=self.transform, use_indices=False
        )
        print("Final test size", len(self.test))
        self.val = DS(
            self.test_df[self.test_df[self.column] != -1], 
            self.column, None, None, transform=self.transform, use_indices=False
        )
        print("Final val size", len(self.val))
        
    def train_dataloader(self):
        """Construct the training set DataLoader"""
        self.train_loader = DataLoader(
            self.train, shuffle=True, batch_size=self.batch_size
        )
        return(self.train_loader)
        
    def test_dataloader(self):
        """Construct the test set DataLoader"""
        self.test_loader = DataLoader(
            self.test, shuffle=False, batch_size=self.batch_size
        )
        return(self.test_loader)
        
    def val_dataloader(self):
        """
        Construct the validation set DataLoader which is 
        the same as the test set because the study design 
        does not require a separate validation set
        """
        self.val_loader = DataLoader(
            self.val, shuffle=False, batch_size=self.batch_size
        )
        return(self.val_loader)


