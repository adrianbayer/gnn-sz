#------------------------------------------------
# Choose parameters to train the networks
# Author: Pablo Villanueva Domingo
# Last update: 5/11/21
#------------------------------------------------

# Choose the GNN architecture between "DeepSet", "GCN", "EdgeNet", "PointNet", "MetaNet"
#use_model = "DeepSet"
#use_model = "GCN"
use_model = "EdgeNet"
#use_model = "PointNet"
#use_model = "EdgePoint"
#use_model = "MetaNet"

# Learning rate
learning_rate = 0.00010992156998246198
# Weight decay
weight_decay = 3.840148429018425e-07
# Number of layers of each graph layer
n_layers = 1
# Number of nearest neighbors in kNN / radius of NNs
k_nn = 100 #  0.14233421449747316 bs is 75,000

# Number of epochs
n_epochs = 150
# If training, set to True, otherwise loads a pretrained model and tests it
training = True
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
#simsuite = "SIMBA"
simsuite = "Quijote"
# Simulation set, choose between "CV" and "LH"
simset = ""
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 5  # maybe will need more later (used 27 if other projects) (need more than one for train/valid/test/)

params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims]
