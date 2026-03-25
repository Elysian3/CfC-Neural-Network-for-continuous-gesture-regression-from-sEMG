import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC