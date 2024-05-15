import numpy as np
import scipy
import random
import scipy.ndimage
from scipy import signal
from scipy.io import loadmat, savemat
import json
import pickle
import numba as nb
import sys
from typing import Literal, Any
from brpylib import NsxFile
import IPython 
import matplotlib.pyplot as plt
from tabulate import tabulate

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

CH_TO_ELECTRODES = [62,  63,  61,  60,  58,  57,  59,  53,  56,  49,  52,  48,  51,
                    44,  54,  43,  55,  38,  50,  42,  45,  37,  47,  36,  46,  35,
                    41,  34,  40,  33,  39,  32,  95,  89,  94,  88,  93,  87,  92,
                    86,  91,  81,  85,  80,  90,  76,  84,  82,  83,  77,  79,  72,
                    78,  73,  74,  75,  70,  71,  67,  68,  65,  69,  64,  66,  127,
                    119, 126, 118, 125, 117, 124, 116, 123, 115, 122, 114, 121, 113,
                    120, 112, 111, 110, 108, 109, 106, 107, 105, 104, 103, 102, 101,
                    100, 99,  98,  96,  97,  31,  29,  30,  28,  27,  26,  25,  24,
                    23,  22,  21,  20,  19,  18,  17,  15,  16,  6,   14,  5,   13,
                    4,   12,  3,   11,  2,   10,  1,   9,   0,   8,   7    ]