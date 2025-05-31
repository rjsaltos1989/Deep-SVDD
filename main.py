#----------------------------------------------------------------------------------
# Deep SVDD
#----------------------------------------------------------------------------------
# Author: Ruff et al. 2018
# Implementation: Ramiro Saltos Atiencia
# Date: 2025-05-15
# Version: 1.5
#----------------------------------------------------------------------------------

# Libraries
#----------------------------------------------------------------------------------

import os
import scipy.io as sio

from torch.utils.data import *
from torch import nn
from nn_models import AutoEncoder, DeepSVDD
from nn_train_functions import *
from svdd_nn_train_functions import *
from plot_functions import *
from svdd_eval_functions import *

# Setting up the device
#device = torch.device("mps" if torch.backends.mps.is_available()
#                      else "cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# %%Importing the data
#----------------------------------------------------------------------------------
dataset_file = 'TwoMoons12D.mat'
data_path = os.path.join('data', dataset_file)


# Load data
mat_data = sio.loadmat(data_path)
data = mat_data['Data']
labels = mat_data['y'].ravel() == 2

# Data dimensionality
num_obs, in_dim = data.shape

# Sphere penalty parameter
nu = 0.05

# %%Data Preparation
#----------------------------------------------------------------------------------

# Create a TensorDataset using the data
train_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                              torch.tensor(data, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                             torch.tensor(labels, dtype=torch.float32))

# Create a DataLoader for each dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%Model Configuration
#----------------------------------------------------------------------------------
latent_dim = 2
layer_sizes = [in_dim, 10, 8, 4, latent_dim]
ae_model = AutoEncoder(layer_sizes)
d_svdd_model = DeepSVDD(ae_model.encoder)
ae_loss_fn = nn.MSELoss()

# %%Model Training
#----------------------------------------------------------------------------------

# Set the max epochs for pretraining and training
pretrain_epochs = 10
train_epochs = 100

# Register the start time
start_time = time.time()

# Run the pretraining phase
results_ae_pd = train_ae_network(ae_model, ae_loss_fn, train_loader, epochs=pretrain_epochs, device=device)

# Initialize the D-SVDD network with the AE encoder weights
d_svdd_model.set_init_weights(ae_model)

# Run the training phase
results_svdd_pd, sph_center, sph_radius = train_d_svdd_network(d_svdd_model, deep_svdd_loss, train_loader, nu=nu,
                                       epochs=train_epochs, device=device)

# Register the end time
end_time = time.time()

print(f"Total training time was {end_time - start_time:.2f} seconds.")
print(f"Threads de OpenMP: {torch.get_num_threads()}")

# %%Plot the results
#----------------------------------------------------------------------------------

plot_training_loss(results_svdd_pd)
plot_d_svdd(data, d_svdd_model, sph_center, sph_radius, device)

# %%Evaluate the performance
#----------------------------------------------------------------------------------

out_scores = get_outlier_scores(d_svdd_model, test_loader, sph_center, device)
eval_metrics = eval_d_svdd(out_scores, labels)
print(eval_metrics)
