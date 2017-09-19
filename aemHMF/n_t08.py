"""
This contains the model for n_t08, or our model for the Tinker et al. 2008 mass function that uses GPs for the fitting function parameters.
"""
import george
import inspect
import numpy as np
import os
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/data_files/"
R_matrix_path = data_path+"R_matrix.txt"
means_path    = data_path+"rotated_dfg_means.txt"
vars_path     = data_path+"rotated_dfg_vars.txt"
