import argparse
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score, MulticlassMatthewsCorrCoef
import numpy as np
import warnings
import os
from torch.utils.data import Dataset, DataLoader 
from collections import OrderedDict
import copy
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pds
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, OrdinalEncoder
import itertools
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from packaging import version
# import weka.core.jvm as jvm
# from weka.classifiers import Classifier
# from weka.core.dataset import Attribute, Instances, Instance
from datetime import datetime
from sklearn.metrics import matthews_corrcoef as mcc, balanced_accuracy_score as bacc,  cohen_kappa_score as kappa, f1_score as f1
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn import over_sampling, under_sampling, combine
from imblearn import ensemble
from joblib import dump, load
import random
import math
warnings.filterwarnings("ignore", category=DeprecationWarning)