"""The module with definitions of the models.
This file provides the definitions to construct every model.
This file is to be imported as a module and exports the following:
    
    NL - A model for Negative Learning
        Restricted Boltzmann Machine
    ABC - A model for Autoencoding Binary Classifier
    CapsNet4AD_Prototype - A prototype of Capsule Network 
        for Anomaly Detection
    CapsNet4HAM - A Capsule Network for HAM10000 dataset
    CapsNet4AD - A Capsule Network for other datasets
    onehot - Performs a one-hot encoding 
        of nucleotide sequence
    correct_order - Reshapes the one-hot 
        encoding array u to (4, length of string)
"""


import torch
import einops
import numpy as np
from torch.optim import Adam
import pytorch_lightning as pl
from capsules.capsules import *
from torchvision import models, transforms
from torch.nn import Conv2d, ReLU, ELU, LeakyReLU, Conv1d
from torch.nn import Sequential, Linear, Tanh, Sigmoid, MSELoss


class NL(pl.LightningModule):
    """
    A model for Negative Learning - Restricted Boltzmann Machine.
    Adapted from https://github.com/eugenet12/pytorch-rbm-autoencoder
    -----------------------------------------------------------------
    ...
    Attributes
    ----------
    momentum_coef : float
        coefficient for momentum
    weight_decay : float
        coefficient for weight decay
    W : Tensor of (size v_size, h_size)
        weights for the RBM
    h_bias : Tensor
        bias for the hidden layer
    v_bias : Tensor
        bias for the visible layer
    W_momentum : Tensor
        momentum for weights
    h_bias_momentum : Tensor
        momentum for hidden layer
    v_bias_momentum : Tensor
        momentum for visible layer
    k : int
        number of Gibbs samples
    mse : MSELoss
        class to compute the loss
    lr : float
        learning rate
            
    Methods
    -------
    sample_h(self, v)  
        Gibbs sampling for hidden layer
    sample_v(self, h)
        Gibbs sampling for visible layer
    update(self, v0, vk, ph0, phk, sign)
        Update the weights according to the CD formula
    forward(self, x)
        Computes forward pass for the RBM
    training_step(self, train_batch, batch_idx)
        Computes training step for the RBM with a positive
        learning phase and negative learning phase
    validation_step(self, val_batch, batch_idx)
        Computes validation step for the RBM
    """
    
    def __init__(
        self, input_shape=(1,28,28), 
        h_size=500, momentum_coef=0.5, weight_decay=2e-4,  k=1, lr=0.01
    ):
        super(NL, self).__init__()
        v_size = np.product(input_shape)
        self.momentum_coef = momentum_coef
        self.weight_decay = weight_decay
        self.W = torch.randn(v_size, h_size)
        self.h_bias = torch.zeros(h_size)
        self.v_bias = torch.zeros(v_size)
        self.W_momentum = torch.zeros(v_size, h_size)
        self.h_bias_momentum = torch.zeros(h_size)
        self.v_bias_momentum = torch.zeros(v_size)
        self.k = k
        self.mse = MSELoss(reduction="mean")
        self.lr = lr
        
    def sample_h(self, v):
        """
        Gibbs sampling for hidden layer.
        v - results of visible layer.
        Returns results of hidden layer
        """
        p = torch.sigmoid(v @ self.W + self.h_bias)
        return(p, torch.bernoulli(p))
    
    def sample_v(self, h):
        """
        Gibbs sampling for visible layer.
        h - results of hidden layer.
        Returns results of visible layer.
        """
        return(torch.sigmoid(h @ self.W.t() + self.v_bias))
    
    def update(self, v0, vk, ph0, phk, sign):
        """
        Updates the weights according to the CD formula.
        v0 - previous results of visible layer.
        vk - new results of visible layer.
        ph0 - previous results of hidden layer.
        phk - new results of hidden layer.
        sign - the sign, 1 for positive learning, -1 for negative learning.
        """
        self.W_momentum *= self.momentum_coef
        self.W_momentum += sign*(v0.t() @ ph0 - vk.t() @ phk)
        self.h_bias_momentum *= self.momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)
        self.v_bias_momentum *= self.momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)
        self.W += self.lr*self.W_momentum/v0.shape[0]
        self.h_bias += self.lr*self.h_bias_momentum/v0.shape[0]
        self.v_bias += self.lr*self.v_bias_momentum/v0.shape[0]
        self.W -= self.W * self.weight_decay
        
    def forward(self, x):
        """
        Computes forward pass for the RBM.
        x - input data.
        Outputs values of visible and hidden layer.
        """
        v0, pvk = x, x
        for i in range(self.k):
            _, hk = self.sample_h(pvk)
            pvk = self.sample_v(hk)
        ph0, _ = self.sample_h(v0)
        phk, _ = self.sample_h(pvk)
        pvk = pvk.squeeze()
        ph0 = ph0.squeeze()
        phk = phk.squeeze()
        if len(ph0.shape) == 1:
            pvk = pvk.reshape(1, -1)
            ph0 = ph0.reshape(1, -1)
            phk = phk.reshape(1, -1)
        return(pvk, ph0, phk)
    
    def training_step(self, train_batch, batch_idx):
        """
        Computes training step for the RBM with a positive
        learning phase and negative learning phase.
        Outputs positive and negative losses.
        """
        x, y = train_batch
        x = x.reshape(x.shape[0], -1).squeeze()
        p_loss = 1
        if x[y == 0].shape[0] != 0:
            pvk, ph0, phk = self(x[y == 0])
            self.update(
                x[y == 0], pvk, ph0, phk, sign=1
            )
            p_loss = self.mse(x[y == 0], pvk)
        n_loss = 1
        if x[y == 1].shape[0] != 0:
            pvk, ph0, phk = self(x[y == 1])
            self.update(
                x[y == 1], pvk, ph0, phk, sign=-1
            )
            n_loss = self.mse(x[y == 1], pvk)
        return(p_loss, n_loss)
    
    def validation_step(self, val_batch, batch_idx):
        """
        Computes validation step for the RBM.
        Outputs positive and negative losses.
        """
        x, y = val_batch
        x = x.reshape(x.shape[0], -1).squeeze()
        pvk, ph0, phk = self(x[y == 0])
        p_loss = self.mse(x[y == 0], pvk)
        pvk, ph0, phk = self(x[y == 1])
        n_loss = self.mse(x[y == 1], pvk)
        return(p_loss, n_loss)


class ABC(pl.LightningModule):
    """
    A model for Autoencoding Binary Classifier
    ------------------------------------------
    ...
    Attributes
    ----------
    encoder : torch.nn.Sequential
        module for the encoding part
    decoder : torch.nn.Sequential
        module for the decoding part
    mse : MSELoss
        module for computing the loss
            
    Methods
    -------
    configure_optimizers(self)
        Configures the optimizer for the ABC
    forward(self, x)
        Computes forward pass for the ABC
    loss(self, x, x_hat, y)
        Computes the ABC loss
    training_step(self, train_batch, batch_idx)
        Computes training step for the ABC
    validation_step(self, val_batch, batch_idx)
        Computes validation step for the ABC
    """
    
    def __init__(self, input_shape=(1,28,28), hiddens=[300, 100], z_size=20):
        super(ABC, self).__init__()
        self.encoder = Sequential(
            Linear(np.product(input_shape), hiddens[0]),
            Tanh(),
            Linear(hiddens[0], hiddens[1]),
            Tanh(),
            Linear(hiddens[1], z_size),
            Tanh()
        )
        self.decoder = Sequential(
            Linear(z_size, hiddens[1]),
            Tanh(),
            Linear(hiddens[1], hiddens[0]),
            Tanh(),
            Linear(hiddens[0], np.product(input_shape)),
            Sigmoid(),
        )
        self.mse = MSELoss(reduction="none")
        
    def forward(self, x):
        """
        Computes forward pass for the ABCs.
        x - input data.
        Outputs the reconstruction.
        """
        ss = x.shape
        z = self.encoder(x.reshape(x.shape[0], -1))
        x_hat = self.decoder(z).reshape(*ss)
        return(x_hat)
    
    def configure_optimizers(self):
        """Configures the optimizer (default Adam)."""
        optimizer = Adam(self.parameters())
        return(optimizer)
    
    def loss(self, x, x_hat, y):
        """Computes the ABC loss."""
        mse = self.mse(x_hat, x).mean(1)
        out = y*mse-(1-y)*torch.log(1-torch.exp(-mse))
        return(out.sum())
    
    def training_step(self, train_batch, batch_idx):
        """
        Computes validation step for the ABC.
        Outputs the loss.
        """
        x, y = train_batch
        x_hat = self(x)
        loss = self.loss(
            x.reshape(x.shape[0], -1), 
            x_hat.reshape(x.shape[0], -1), y
        )
        self.log("train_loss", loss)
        return(loss)
    
    def validation_step(self, val_batch, batch_idx):
        """
        Computes validation step for the ABC.
        Outputs the loss.
        """
        x, y = val_batch
        x_hat = self(x)
        loss = self.loss(
            x.reshape(x.shape[0], -1), 
            x_hat.reshape(x.shape[0], -1), y
        )
        self.log("val_loss", loss)
        return(loss)


class CapsNet4AD_Prototype(pl.LightningModule):
    """
    A prototype of Capsule Network for Anomaly Detection
    ----------------------------------------------------
    ...
    Attributes
    ----------
    No attributes - they are to be defined 
        in the descendant classes
            
    Methods
    -------
    configure_optimizers(self)
        Configures the optimizer for the CapsNet
    forward(self, x)
        Computes forward pass for the CapsNet
    capsule_loss(self, real_class, x, classes, reconstruction)
        Computes the margin loss
    training_step(self, train_batch, batch_idx)
        Computes training step for the CapsNet
    validation_step(self, val_batch, batch_idx)
        Computes validation step for the CapsNet
    """
    
    def __init__(self):
        super(CapsNet4AD_Prototype, self).__init__()
    
    def forward(self, x):
        """Compute forward of capsules, get the longest vectors, 
        reconstruct the pictures"""
        u = self.conv(x)
        u = self.primcaps(u)
        internal, a = self.digicaps(u)
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked = torch.eye(self.n_classes)
        masked = masked.cuda() if torch.cuda.is_available() else masked
        masked = masked.index_select(dim=0, index=max_caps_index)
        reconstruction = self.decoder(
            (internal*masked[:,:,None]).reshape(x.size(0), -1)
        )
        return(internal, reconstruction, lengths, max_caps_index, a)
    
    def configure_optimizers(self):
        """Configures the optimizer (default Adam)."""
        optimizer = Adam(self.parameters())
        return(optimizer)
    
    def capsule_loss(self, real_class, x, classes, reconstruction):
        """Computes margin loss"""
        return(
            self.capsule_loss_object(
                real_class, x.reshape(x.shape[0], -1),
                classes, reconstruction
            )
        )
    
    def training_step(self, train_batch, batch_idx):
        """
        Computes training step for the CapsNet.
        Outputs the loss.
        """
        x, y = train_batch
        real_class = make_y(y.type(torch.LongTensor).cuda(), self.n_classes)
        internal, reconstruction, classes, max_index, _ = self.forward(x)
        loss = self.capsule_loss(
            real_class, x, classes, reconstruction
        )
        self.log("train_loss", loss)
        return(loss)
    
    def validation_step(self, val_batch, batch_idx):
        """
        Computes validation step for the CapsNet.
        Outputs the loss.
        """
        x, y = val_batch
        real_class = make_y(y.type(torch.LongTensor).cuda(), self.n_classes)
        internal, reconstruction, classes, max_index, _ = self.forward(x)
        loss = self.capsule_loss(
            real_class, x, classes, reconstruction
        )
        self.log("val_loss", loss)
        return(loss)


class CapsNet4HAM(CapsNet4AD_Prototype):
    """
    A Capsule Network for HAM10000 dataset
    --------------------------------------
    ...
    Attributes
    ----------
    conv : torch.nn.Sequential
        A module for the convolutional 
        preprocessing
    n_classes : int
        number of classes
    primcaps : Primary Capsule layer
    digicaps : Secondary Capsule layer
    decoder : Regularizing decoder
    capsule_loss_object : the object for 
        Margin Loss
            
    Methods
    -------
    No methods but the ones that 
        were defined in prototype
    """
    
    def __init__(self, n_primary: int = 20736, rec_size: int = 22500, n_classes: int = 2):
        super().__init__()
        self.conv = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=9,
                stride=1
            ),
            ELU(inplace=True)
        )
        self.n_classes = n_classes
        self.primcaps = PrimaryCapsuleLayer()
        self.digicaps = SecondaryCapsuleLayer(
            n_capsules=n_classes, return_couplings=True, n_primary=n_primary
        )
        self.decoder = RegularizingDecoder(dims=[32,512,1024,rec_size])
        self.capsule_loss_object = CapsuleLoss(
            only_normals=True, normal_class=0
        )


class CapsNet4AD(CapsNet4AD_Prototype):
    """
    A Capsule Network for other datasets
    ------------------------------------
    ...
    Attributes
    ----------
    conv : torch.nn.Sequential
        A module for the convolutional 
        preprocessing
    n_classes : int
        number of classes
    primcaps : Primary Capsule layer
    digicaps : Secondary Capsule layer
    decoder : Regularizing decoder
    capsule_loss_object : the object for
        Margin Loss
            
    Methods
    -------
    No methods but the ones that 
        were defined in prototype
    """
    
    def __init__(
        self, n_primary: int = 1152, in_channels: int = 1, 
        rec_size: int = 28*28, kernel_size: int = 9, primcaps_kernel_size: int = 9,
        n_classes: int = 2
    ):
        super().__init__()
        self.conv = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=kernel_size,
                stride=1
            ),
            ReLU(inplace=True)
        )
        self.primcaps = PrimaryCapsuleLayer(
            n_convs=32, out_ch=32, kernel_size=primcaps_kernel_size
        )
        self.digicaps = SecondaryCapsuleLayer(
            n_capsules=n_classes, return_couplings=True, n_primary=n_primary,
            in_ch=32
        )
        self.n_classes = n_classes
        self.decoder = RegularizingDecoder(dims=[16*n_classes,512,1024,rec_size])
        self.capsule_loss_object = CapsuleLoss(
            only_normals=True, normal_class=0
        )


def onehot(u):
    """
    Performs a one-hot encoding of nucleotide sequence
    
    u is the input string
    
    Outputs a 1D numpy array with the encoding 
    """
    encoding = {
        1: [1,0,0,0],
        2: [0,0,0,1],
        3: [0,1,0,0],
        4: [0,0,1,0],
        0: [0,0,0,0],
        "A": [1,0,0,0],
        "T": [0,0,0,1],
        "G": [0,1,0,0],
        "C": [0,0,1,0],
        "N": [0,0,0,0],
        "a": [1,0,0,0],
        "t": [0,0,0,1],
        "g": [0,1,0,0],
        "c": [0,0,1,0],
        "n": [0,0,0,0]
    }
    r = np.array(sum([encoding[a] for a in u], []))
    return(r)


def correct_order(u, k=4):
    """Reshapes the one-hot encoding array u to (4, length of string)"""
    return(
        u.reshape((k,int(u.shape[0]/k)), order="f").reshape(u.shape[0])
    )
