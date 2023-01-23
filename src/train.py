import config as cfg
import engine
from utils import seed_everything
from dataset import TrainDataset
from model import Deberta_v3_large_LSTM

import os
import re
import gc
import math
from tqdm import tqdm


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup


if __name__  == "__main__":

    
    seed_everything(42)
    train = pd.read_csv(CFG.train_data_path)
    print(f"train.shape: {train.shape}")
    display(train.head())
