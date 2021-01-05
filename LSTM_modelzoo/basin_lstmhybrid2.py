# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:03:55 2019

@author: 5gk
"""
import os
from pathlib import Path
from typing import Tuple, List
import lstmhyd as lt
from numba import njit
import numpy as np
import pandas as pd
import glob
from mpi4py import MPI
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
CAMELS_ROOT = Path('/lustre/or-hydra/cades-ccsi/scratch/5gk/lstmsUSA/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2')
CAMELS_MODEL = Path('/lustre/or-hydra/cades-ccsi/scratch/5gk/lstmsUSA/basin_timeseries_v1p2_modelOutput_daymet/model_output_daymet/model_output')
DEVICE = lt.torch.device("cpu") # This line checks if GPU is available

#basinlist = ['09034900','09035800','09035900','09047700','09065500','09066000','09066200','09066300','09081600','09107000','09210500','09223000','09306242','09312600','09352900','09378170','09378630'] # can be changed to any 8-digit basin id contained in the CAMELS data set
#basinlist = ['09378170','09107000']
hidden_size = 20 # Number of LSTM cells
dropout_rate = 0.1 # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 1e-3 # Learning rate used to update the weights
sequence_length = 365 # Length of the meteorological record provided to the network
basinlist = ['09034900']
##############
# Data set up#
##############

# Training data

basinlist = []
f=open('/lustre/or-hydra/cades-ccsi/scratch/5gk/lstmsUSA/basin_list2.txt',"r")
for line in f:
    basinlist.append(line.strip('\n'))
		

def test_te(i):
    lt.torch.manual_seed(0)
    basin = basinlist[i]
    print(basin)
    start_date = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("1996-09-30", format="%Y-%m-%d")
    ds_train = lt.CamelsTXT(basin, seq_length=sequence_length, period="train", dates=[start_date, end_date])
    tr_loader = lt.DataLoader(ds_train, batch_size=256, shuffle=True)

# Validation data. We use the feature means/stds of the training period for normalization
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    start_date = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2011-09-30", format="%Y-%m-%d")
    ds_val = lt.CamelsTXT(basin, seq_length=sequence_length, period="eval", dates=[start_date, end_date],
                     means=means, stds=stds)
    val_loader = lt.DataLoader(ds_val, batch_size=2048, shuffle=False)

# Test data. We use the feature means/stds of the training period for normalization



#########################
# Model, Optimizer, Loss#
#########################

# Here we create our model, feel free
    model = lt.Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = lt.torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
    loss_func = lt.NSELoss()

    n_epochs = 100 # Number of training epochs
    validnselist = []
    fdc_slope = []
    fdc_low = []
    fdc_high = []

    for i in range(n_epochs):
        lt.train_epoch(model, optimizer, tr_loader, loss_func, i+1)
        obs, preds = lt.eval_model(model, val_loader)
        preds = ds_val.local_rescale(preds.numpy(), variable='output')
        validationnse = lt.calc_nse(obs.numpy(), preds)
        validnselist.append(validationnse)
        print(validationnse)
        obs = obs.numpy()
        fdc_slope.append(lt.calc_fdc_fms(obs,preds))
        fdc_low.append(lt.calc_fdc_flv(obs,preds))
        fdc_high.append(lt.calc_fdc_fhv(obs,preds))

    with open(f"{basin}_NSE_hybrid1_baseline.txt",'w') as f:
        for item in validnselist:
            f.write("%s\n" % item)

    with open(f"{basin}_fdc_slope_hybrid1_baseline.txt",'w') as f:
        for item in fdc_slope:
            f.write("%s\n" % item)

    with open(f"{basin}_fdc_low_hybrid1_baseline.txt",'w') as f:
        for item in fdc_low:
            f.write("%s\n" % item)

    with open(f"{basin}_fdc_high_hybrid1_baseline.txt",'w') as f:
        for item in fdc_high:
            f.write("%s\n" % item)



task_list = range(211)


for i,task in enumerate(task_list):
  #This is how we split up the jobs.
  #The % sign is a modulus, and the "continue" means
  #"skip the rest of this bit and go to the next time
  #through the loop"
  # If we had e.g. 4 processors, this would mean
  # that proc zero did tasks 0, 4, 8, 12, 16, ...
  # and proc one did tasks 1, 5, 9, 13, 17, ...
  # and do on.
    if i%size!=rank: continue
    print("Task number %d (%d) being done by processor %d of %d" % (i, task, rank, size))
    test_te(task)
