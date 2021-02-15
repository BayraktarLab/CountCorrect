# -*- coding: utf-8 -*-
"""Base model class"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from plot_heatmap import clustermap

# base model class - defining shared methods but not the model itself
class BaseModel():
    r"""Base class for pymc3 and pyro models.

    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param n_fact: Number of factors
    :param n_iter: Number of training iterations
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param data_type: theano data type used to store parameters ('float32' for single, 'float64' for double precision)
    :param total_grad_norm_constraint: gradient constraints in optimisation
    :param verbose: print diagnostic messages?
    :param var_names: Variable names (e.g. gene identifiers)
    :param var_names_read: Readable variable names (e.g. gene symbol)
    :param obs_names: Observation names (e.g. cell or spot id)
    :param fact_names: Factor names
    :param sample_id: Sample identifiers (e.g. different experiments)
    """

    def __init__(
            self,
            X_data: np.ndarray,
            n_fact: int = 10,
            data_type: str = 'float32',
            n_iter: int = 200000,
            learning_rate=0.001,
            total_grad_norm_constraint=200,
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None
    ):

        # Initialise parameters
        self.X_data = X_data
        self.n_fact = n_fact
        self.n_genes = X_data.shape[1]
        self.n_cells = X_data.shape[0]
        self.data_type = data_type
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.total_grad_norm_constraint = total_grad_norm_constraint
        self.verbose = verbose
        self.fact_filt = None
        self.gene_loadings = None
        self.minibatch_size = None
        self.minibatch_seed = None
        self.extra_data = None  # input data
        self.extra_data_tt = None  # minibatch parameters

        # add essential annotations
        if var_names is None:
            self.var_names = pd.Series(['g_' + str(i) for i in range(self.n_genes)],
                                       index=['g_' + str(i) for i in range(self.n_genes)])
        else:
            self.var_names = pd.Series(var_names,
                                       index=var_names)

        if var_names_read is None:
            self.var_names_read = pd.Series(self.var_names, index=self.var_names)
        else:
            self.var_names_read = pd.Series(var_names_read, index=self.var_names)

        if obs_names is None:
            self.obs_names = pd.Series(['c_' + str(i) for i in range(self.n_cells)],
                                       index=['c_' + str(i) for i in range(self.n_cells)])
        else:
            self.obs_names = pd.Series(obs_names, index=obs_names)

        if fact_names is None:
            self.fact_names = pd.Series(['fact_' + str(i) for i in range(self.n_fact)])
        else:
            self.fact_names = pd.Series(fact_names)

        if sample_id is None:
            self.sample_id = pd.Series(['sample' for i in range(self.n_cells)],
                                       index=self.obs_names)
        else:
            self.sample_id = pd.Series(sample_id, index=self.obs_names)

    def plot_history(self, iter_start=0, iter_end=-1, log_y=True):
        r""" Plot training history

        :param iter_start: omit initial iterations from the plot
        :param iter_end: omit last iterations from the plot
        """
        for i in self.hist.keys():
            y = self.hist[i][iter_start:iter_end]
            if log_y:
                y = np.log10(y)

            plt.plot(y)
            plt.tight_layout()
