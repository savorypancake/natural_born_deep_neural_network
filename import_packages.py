# import the required packages

import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
# from testCases import *
#from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
# from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)