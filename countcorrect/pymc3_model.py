# -*- coding: utf-8 -*-
r"""Base Pymc3 model class for all models in pymc3"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
from pymc3.variational.callbacks import CheckParametersConvergence
from tqdm.auto import tqdm

from base_model import BaseModel

# base model class - defining shared methods but not the model itself
class Pymc3Model(BaseModel):
    r"""This class provides functions to train PyMC3 models and sample their parameters.
    A model must have a main X_data input and can have arbitrary self.extra_data inputs.

    Parameters
    ----------
    X_data :
        Numpy array of gene expression (cols) in spatial locations (rows)
    all other:
        the rest are arguments for parent class BaseModel

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

        ############# Initialise parameters ################
        super().__init__(X_data, n_fact,
                         data_type, n_iter,
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)

        # Create dictionaries storing results
        self.advi = {}
        self.mean_field = {}
        self.samples = {}
        self.node_samples = {}
        self.n_type = 'restart' # default

        # Pass data to theano
        self.x_data = theano.shared(X_data.astype(self.data_type))

    def sample_prior(self, samples=10):
        r"""Take samples from the prior, see `pymc3.sample_prior_predictive` for details

        Parameters
        ----------
        samples :
             (Default value = 10)

        Returns
        -------
        dict
            self.prior_trace dictionary with an element for each parameter of the mode
        """
        # Take samples from the prior
        with self.model:
            self.prior_trace = pm.sample_prior_predictive(samples=samples)

    def fit_advi(self, n=3, method='advi', n_type='restart'):
        r"""Find posterior using ADVI (maximising likehood of the data and
        minimising KL-divergence of posterior to prior)

        Parameters
        ----------
        n :
            number of independent initialisations (Default value = 3)
        method :
            advi', to allow for potential use of SVGD, MCMC, custom (currently only ADVI implemented). (Default value = 'advi')
        n_type :
            type of repeated initialisation:

            * **'restart'** to pick different initial value,
            * **'cv'** for molecular cross-validation - splits counts into n datasets, for now, only n=2 is implemented
            * **'bootstrap'** for fitting the model to multiple downsampled datasets.
              Run `mod.bootstrap_data()` to generate variants of data (Default value = 'restart')

        Returns
        -------
        dict
            self.mean_field dictionary with MeanField pymc3 objects for each initialisation.

        """

        if not np.isin(n_type, ['restart', 'cv', 'bootstrap']):
            raise ValueError("n_type should be one of ['restart', 'cv', 'bootstrap']")

        self.n_type = n_type

        if np.isin(n_type, ['bootstrap']):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ['cv']):
            self.generate_cv_data(n=n)  # cv data added to self.X_data_sample

        init_names = ['init_' + str(i + 1) for i in np.arange(n)]

        with self.model:

            for i, name in enumerate(init_names):

                # when type is molecular cross-validation or bootstrap, 
                # replace self.x_data tensor with new data
                if np.isin(n_type, ['cv', 'bootstrap']):
                    more_replacements = {self.x_data: self.X_data_sample[i].astype(self.data_type)}
                else:
                    more_replacements = {}

                # train the model
                self.mean_field[name] = pm.fit(self.n_iter, method='advi',
                                               callbacks=[CheckParametersConvergence()],
                                               obj_optimizer=pm.adam(learning_rate=self.learning_rate),
                                               total_grad_norm_constraint=self.total_grad_norm_constraint,
                                               more_replacements=more_replacements)

                # plot training history
                if self.verbose:
                    plt.plot(np.log10(self.mean_field[name].hist[15000:]))

    def fit_advi_iterative(self, n=3, method='advi', n_type='restart',
                           n_iter=None,
                           learning_rate=None, reducing_lr=False,
                           progressbar=True,
                           scale_cost_to_minibatch=True):
        """Find posterior using pm.ADVI() method directly (allows continuing training through `refine` method.
        (maximising likelihood of the data and minimising KL-divergence of posterior to prior - ELBO loss)

        Parameters
        ----------
        n :
            number of independent initialisations (Default value = 3)
        method :
            advi', to allow for potential use of SVGD, MCMC, custom (currently only ADVI implemented). (Default value = 'advi')
        n_type :
            type of repeated initialisation:
            
            * **'restart'** to pick different initial value,
            * **'cv'** for molecular cross-validation - splits counts into n datasets, for now, only n=2 is implemented
            * **'bootstrap'** for fitting the model to multiple downsampled datasets.
              Run `mod.bootstrap_data()` to generate variants of data (Default value = 'restart')

        n_iter :
            number of iterations, supersedes self.n_iter specified when creating model instance. (Default value = None)
        learning_rate :
            learning rate, supersedes self.learning_rate specified when creating model instance. (Default value = None)
        reducing_lr :
            boolean, use decaying learning rate? (Default value = False)
        progressbar :
            boolean, show progress bar? (Default value = True)
        scale_cost_to_minibatch :
            when using training in minibatches, scale cost function appropriately?
            See discussion https://discourse.pymc.io/t/effects-of-scale-cost-to-minibatch/1429 to understand the effects. (Default value = True)

        Returns
        -------
        None
            self.mean_field dictionary with MeanField pymc3 objects,
            and self.advi dictionary with ADVI objects for each initialisation.

        """

        self.n_type = n_type
        self.scale_cost_to_minibatch = scale_cost_to_minibatch

        if n_iter is None:
            n_iter = self.n_iter

        if learning_rate is None:
            learning_rate = self.learning_rate

        ### Initialise optimiser ###
        if reducing_lr:
            # initialise the function for adaptive learning rate
            s = theano.shared(np.array(learning_rate).astype(self.data_type))

            def reduce_rate(a, h, i):
                s.set_value(np.array(learning_rate / ((i / self.n_cells) + 1) ** .7).astype(self.data_type))

            optimiser = pm.adam(learning_rate=s)
            callbacks = [reduce_rate, CheckParametersConvergence()]
        else:
            optimiser = pm.adam(learning_rate=learning_rate)
            callbacks = [CheckParametersConvergence()]

        if np.isin(n_type, ['bootstrap']):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ['cv']):
            self.generate_cv_data()  # cv data added to self.X_data_sample

        init_names = ['init_' + str(i + 1) for i in np.arange(n)]

        for i, name in enumerate(init_names):

            with self.model:

                self.advi[name] = pm.ADVI(random_seed = 99)

            # when type is molecular cross-validation or bootstrap, 
            # replace self.x_data tensor with new data
            if np.isin(n_type, ['cv', 'bootstrap']):

                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(self.X_data_sample[i].astype(self.data_type),
                                                         batch_size=[self.minibatch_size, None],
                                                         random_seed=self.minibatch_seed[i])
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = \
                                pm.Minibatch(self.extra_data[k].astype(self.data_type),
                                             batch_size=[self.minibatch_size, None],
                                             random_seed=self.minibatch_seed[i])

                # or using all data
                else:
                    more_replacements = {self.x_data: self.X_data_sample[i].astype(self.data_type)}
                    # if any other data inputs should be added
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = \
                                self.extra_data[k].astype(self.data_type)

            else:

                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(self.X_data.astype(self.data_type),
                                                         batch_size=[self.minibatch_size, None],
                                                         random_seed=self.minibatch_seed[i])
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = \
                                pm.Minibatch(self.extra_data[k].astype(self.data_type),
                                             batch_size=[self.minibatch_size, None],
                                             random_seed=self.minibatch_seed[i])

                else:
                    more_replacements = {}

            self.advi[name].scale_cost_to_minibatch = scale_cost_to_minibatch

            # train the model  
            self.mean_field[name] = self.advi[name].fit(n_iter, callbacks=callbacks,
                                                        obj_optimizer=optimiser,
                                                        total_grad_norm_constraint=self.total_grad_norm_constraint,
                                                        progressbar=progressbar, more_replacements=more_replacements,
                                                        )

            # plot training history
            if self.verbose:
                print(plt.plot(np.log10(self.mean_field[name].hist[15000:])));

    def fit_advi_refine(self, n_iter=10000, learning_rate=None,
                        progressbar=True, reducing_lr=False):
        """Refine posterior using ADVI - continue training after `.fit_advi_iterative()`

        Parameters
        ----------
        n_iter :
            number of additional iterations (Default value = 10000)
        learning_rate :
            same as in `.fit_advi_iterative()` (Default value = None)
        progressbar :
            same as in `.fit_advi_iterative()` (Default value = True)
        reducing_lr :
            same as in `.fit_advi_iterative()` (Default value = False)

        Returns
        -------
        dict
            update the self.mean_field dictionary with MeanField pymc3 objects.

        """

        self.n_iter = self.n_iter + n_iter

        if learning_rate is None:
            learning_rate = self.learning_rate

        ### Initialise optimiser ###
        if reducing_lr:
            # initialise the function for adaptive learning rate
            s = theano.shared(np.array(learning_rate).astype(self.data_type))

            def reduce_rate(a, h, i):
                s.set_value(np.array(learning_rate / ((i / self.n_cells) + 1) ** .7).astype(self.data_type))

            optimiser = pm.adam(learning_rate=s)
            callbacks = [reduce_rate, CheckParametersConvergence()]
        else:
            optimiser = pm.adam(learning_rate=learning_rate)
            callbacks = [CheckParametersConvergence()]

        for i, name in enumerate(self.advi.keys()):

            # when type is molecular cross-validation or bootstrap,
            # replace self.x_data tensor with new data
            if np.isin(self.n_type, ['cv', 'bootstrap']):

                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(self.X_data_sample[i].astype(self.data_type),
                                                         batch_size=[self.minibatch_size, None],
                                                         random_seed=self.minibatch_seed[i])
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = \
                                pm.Minibatch(self.extra_data[k].astype(self.data_type),
                                             batch_size=[self.minibatch_size, None],
                                             random_seed=self.minibatch_seed[i])

                # or using all data
                else:
                    more_replacements = {self.x_data: self.X_data_sample[i].astype(self.data_type)}
                    # if any other data inputs should be added
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = \
                                self.extra_data[k].astype(self.data_type)

            else:
                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(self.X_data.astype(self.data_type),
                                                         batch_size=[self.minibatch_size, None],
                                                         random_seed=self.minibatch_seed[i])
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = \
                                pm.Minibatch(self.extra_data[k].astype(self.data_type),
                                             batch_size=[self.minibatch_size, None],
                                             random_seed=self.minibatch_seed[i])

                else:
                    more_replacements = {}

            with self.model:
                # train for more iterations & export trained model by overwriting the initial mean field object
                self.mean_field[name] = self.advi[name].fit(n_iter, callbacks=callbacks,
                                                            obj_optimizer=optimiser,
                                                            total_grad_norm_constraint=self.total_grad_norm_constraint,
                                                            progressbar=progressbar,
                                                            more_replacements=more_replacements)

                if self.verbose:
                    print(plt.plot(np.log10(self.mean_field[name].hist[15000:])))

    def plot_history(self, iter_start=0, iter_end=-1):
        """Plot loss function (ELBO) across training history

        Parameters
        ----------
        iter_start :
            omit initial iterations from the plot (Default value = 0)
        iter_end :
            omit last iterations from the plot (Default value = -1)

        """
        for i in self.mean_field.keys():
            print(plt.plot(np.log10(self.mean_field[i].hist[iter_start:iter_end])))

    def sample_posterior(self, node='all', n_samples=1000,
                         save_samples=False, return_samples=True,
                         mean_field_slot='init_1'):
        """Sample posterior distribution of all parameters or single parameter

        Parameters
        ----------
        node :
            pymc3 node to sample (e.g. default "all", self.spot_factors)
        n_samples :
            number of posterior samples to generate (1000 is recommended, reduce if you get GPU memory error) (Default value = 1000)
        save_samples :
            save all samples, not just the mean, 5% and 95% quantile, SD. (Default value = False)
        return_samples :
            return summarised samples (mean, etc) in addition to saving them in `self.samples`? (Default value = True)
        mean_field_slot :
            string, which training initialisation (mean_field slot) to sample? 'init_1' by default

        Returns
        -------
        dict
            dictionary `self.samples` (mean, 5% quantile, SD, optionally all samples) with dictionaries
            with numpy arrays for each parameter.
            Plus an optional dictionary in `self.samples` with all samples of parameters
            as numpy arrays of shape ``(n_samples, ...)``

        """

        theano.config.compute_test_value = 'ignore'

        if node == 'all':
            # Sample all parameters - might use a lot of GPU memory
            post_samples = self.mean_field[mean_field_slot].sample(n_samples)
            self.samples['post_sample_means'] = {v: post_samples[v].mean(axis=0) for v in post_samples.varnames}
            self.samples['post_sample_q05'] = {v: np.quantile(post_samples[v], 0.05, axis=0) for v in
                                               post_samples.varnames}
            self.samples['post_sample_q95'] = {v: np.quantile(post_samples[v], 0.95, axis=0) for v in
                                               post_samples.varnames}            
            self.samples['post_sample_q01'] = {v: np.quantile(post_samples[v], 0.01, axis=0) for v in
                                               post_samples.varnames}
            self.samples['post_sample_q99'] = {v: np.quantile(post_samples[v], 0.99, axis=0) for v in
                                               post_samples.varnames}
            self.samples['post_sample_sds'] = {v: post_samples[v].std(axis=0) for v in post_samples.varnames}

            if (save_samples):
                # convert multitrace object to a dictionary
                post_samples = {v: post_samples[v] for v in post_samples.varnames}
                self.samples['post_samples'] = post_samples

        else:
            # Sample a singe node
            post_node = self.mean_field[mean_field_slot].sample_node(node, size=n_samples).eval()
            post_node_mean = post_node.mean(0)
            post_node_q05 = np.quantile(post_node, 0.05, axis=0)
            post_node_q95 = np.quantile(post_node, 0.95, axis=0)
            post_node_q01 = np.quantile(post_node, 0.01, axis=0)
            post_node_q99 = np.quantile(post_node, 0.99, axis=0)
            post_node_sds = post_node.std(0)

            # extract the name of the node and save to samples dictionary
            node_name = node.name
            self.node_samples[str(node_name) + '_mean'] = post_node_mean
            self.node_samples[str(node_name) + '_q05'] = post_node_q05
            self.node_samples[str(node_name) + '_q95'] = post_node_q95
            self.node_samples[str(node_name) + '_q01'] = post_node_q01
            self.node_samples[str(node_name) + '_q99'] = post_node_q99
            self.node_samples[str(node_name) + '_sds'] = post_node_sds

            if save_samples:
                self.node_samples[str(node_name) + '_post_samples'] = post_node

        if return_samples:
            return self.samples
