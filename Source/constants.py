import numpy as np
import torch
import os
import random

torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

#--- PARAMETERS ---#

# Root path for simulations
simpathroot = "/projects/QUIJOTE/CAMELS/Sims/"

# Box size in comoving kpc/h
boxsize = 25.e3

# Validation and test size
valid_size, test_size = 0.15, 0.15

# 1 for testing only
only_test = 0
if only_test:   valid_size, test_size = 0.005, 0.99

# Batch size
batch_size = 128

# Weight of the message L1 regularization in the total loss respect to the standard loss
l1_reg = 0.01

data_aug = 1

# 1 if train for performing symbolic regression later, 0 otherwise
sym_reg = 0

# 1 if use L1 regularization with messages. Needed for symbolic regression
use_l1 = 0

# Name of the model and hyperparameters
def namemodel(params):
    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simtype, simset, n_sims = params
    return simtype+"_"+simset+"_model_"+use_model+"_lr_{:.2e}_weightdecay_{:.2e}_layers_{:d}_knn_{:.2e}_epochs_{:d}".format(learning_rate, weight_decay, n_layers, k_nn, n_epochs)