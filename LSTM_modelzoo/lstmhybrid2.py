# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:58:29 2019

@author: 5gk
"""

# Imports
from pathlib import Path
from typing import Tuple, List

from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import glob
torch.manual_seed(0)
# Globals
CAMELS_ROOT = Path('/lustre/or-hydra/cades-ccsi/scratch/5gk/lstmsUSA/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2')
CAMELS_MODEL = Path('/lustre/or-hydra/cades-ccsi/scratch/5gk/lstmsUSA/basin_timeseries_v1p2_modelOutput_daymet/model_output_daymet/model_output')
DEVICE = torch.device("cpu") # This line checks if GPU is available

def load_forcing(basin: str) -> Tuple[pd.DataFrame]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    # root directory of meteorological forcings
    forcing_path = CAMELS_ROOT / 'basin_mean_forcing' / 'daymet'

    # get path of forcing file
    files = list(glob.glob(f"{str(forcing_path)}/**/{basin}_*.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index
    with open(file_path) as fp:
        df = pd.read_csv(fp, sep='\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with open(file_path) as fp:
        content = fp.readlines()
        area = int(content[2])

    return df


def load_discharge(basin: str) ->  pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters

    :return: A pd.Series containng the catchment normalized discharge.
    """
    # root directory of the streamflow data
    model_path = CAMELS_MODEL / 'flow_timeseries' / 'daymet'

    
  # root directory of model output forcings
    
        
    # get path of model output file
    files = list(glob.glob(f"{str(model_path)}/**/{basin}_*model_output.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No Model output file found for Basin {basin}')
    else:
        file_path_out = files[0]
    
    # read-in data and convert date to datetime index
    with open(file_path_out) as fp:
        df_out = pd.read_csv(fp, sep='\s+', header=0)
    dates = (df_out.YR.map(str) + "/" + df_out.MNTH.map(str) + "/"
             + df_out.DY.map(str))
    df_out.index = pd.to_datetime(dates, format="%Y/%m/%d")
    df_out['Residual']=df_out['OBS_RUN']-df_out['MOD_RUN']
    return df_out.Residual 
	
	
	
def load_moddischarge(basin: str) ->  pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters
    
    :return: A pd.Series containng the catchment normalized discharge.
    """
    # root directory of the streamflow data
    model_path = CAMELS_MODEL / 'flow_timeseries' / 'daymet'

    
  # root directory of model output forcings
    
        
    # get path of model output file
    files = list(glob.glob(f"{str(model_path)}/**/{basin}_*model_output.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No Model output file found for Basin {basin}')
    else:
        file_path_out = files[0]
    
    # read-in data and convert date to datetime index
    with open(file_path_out) as fp:
        df_out = pd.read_csv(fp, sep='\s+', header=0)
    dates = (df_out.YR.map(str) + "/" + df_out.MNTH.map(str) + "/"
             + df_out.DY.map(str))
    df_out.index = pd.to_datetime(dates, format="%Y/%m/%d")
    return df_out.MOD_RUN 


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

class CamelsTXT(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """

    def __init__(self, basin: str, seq_length: int=365,period: str=None,
                 dates: List=None, means: pd.Series=None, stds: pd.Series=None):
        """Initialize Dataset containing the data of a single basin.

        :param basin: 8-digit code of basin as string.
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        """
        self.basin = basin
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds

        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df = load_forcing(self.basin)
        df['QObs(mm/d)'] = load_discharge(self.basin)

        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # if training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        # extract input and output features from DataFrame
        x = np.array([df['prcp(mm/day)'].values,
                      df['srad(W/m2)'].values,
                      df['tmax(C)'].values,
                      df['tmin(C)'].values,
                      df['vp(Pa)'].values]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) ->             np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["QObs(mm/d)"]) /
                       self.stds["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) ->             np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds



class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.
    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the
    discharge from the basin, to which the sample belongs.
    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1):
        super(NSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.
        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
               Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """

        numerator = torch.sum((y_pred - y_true)**2)
        denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)
        nse_val = (numerator / denominator) - 1

        return nse_val





class Model(nn.Module):
    """Implementation of a single layer LSTM network"""

    def __init__(self, hidden_size: int, dropout_rate: float=0.0):
        """Initialize model

        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # create required layer
        self.lstm = nn.LSTM(input_size=5, hidden_size=self.hidden_size,
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.

        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.

        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)

        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred




def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm_notebook(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)

def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val

def calc_fdc_fms(obs: np.ndarray, sim: np.ndarray, m1: float = 0.2, m2: float = 0.7) -> float:
    """[summary]

    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    m1 : float, optional
        Lower bound of the middle section. Has to be in range(0,1), by default 0.2
    m2 : float, optional
        Upper bound of the middle section. Has to be in range(0,1), by default 0.2

    Returns
    -------
    float
        Bias of the middle slope of the flow duration curve (Yilmaz 2018).

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `m1` is not in range(0,1)
    RuntimeError
        If `m2` is not in range(0,1)
    RuntimeError
        If `m1` >= `m2`
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (m1 <= 0) or (m1 >= 1):
        raise RuntimeError("m1 has to be in the range (0,1)")

    if (m2 <= 0) or (m2 >= 1):
        raise RuntimeError("m1 has to be in the range (0,1)")

    if m1 >= m2:
        raise RuntimeError("m1 has to be smaller than m2")

    # for numerical reasons change 0s to 1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # calculate fms part by part
    qsm1 = np.log(sim[np.round(m1 * len(sim)).astype(int)] + 1e-6)
    qsm2 = np.log(sim[np.round(m2 * len(sim)).astype(int)] + 1e-6)
    qom1 = np.log(obs[np.round(m1 * len(obs)).astype(int)] + 1e-6)
    qom2 = np.log(obs[np.round(m2 * len(obs)).astype(int)] + 1e-6)

    fms = ((qsm1 - qsm2) - (qom1 - qom2)) / (qom1 - qom2 + 1e-6)

    return fms * 100


def calc_fdc_fhv(obs: np.ndarray, sim: np.ndarray, h: float = 0.02) -> float:
    """Peak flow bias of the flow duration curve (Yilmaz 2018).

    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    h : float, optional
        Fraction of the flows considered as peak flows. Has to be in range(0,1), by default 0.02

    Returns
    -------
    float
        Bias of the peak flows

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `h` is not in range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h has to be in the range (0,1)")

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

    return fhv * 100


def calc_fdc_flv(obs: np.ndarray, sim: np.ndarray, l: float = 0.7) -> float:
    """[summary]

    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    l : float, optional
        Upper limit of the flow duration curve. E.g. 0.7 means the bottom 30% of the flows are
        considered as low flows, by default 0.7

    Returns
    -------
    float
        Bias of the low flows.

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `l` is not in the range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (l <= 0) or (l >= 1):
        raise RuntimeError("l has to be in the range (0,1)")

    # for numerical reasons change 0s to 1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[np.round(l * len(obs)).astype(int):]
    sim = sim[np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs + 1e-6)
    sim = np.log(sim + 1e-6)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100
