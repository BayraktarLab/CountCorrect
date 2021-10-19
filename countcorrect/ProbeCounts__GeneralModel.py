# -*- coding: utf-8 -*-
"""Generative Model for NanostringWTA counts

This is a more advanced version of the model that includes:

- overdispersion on top of the gene expression model
- a minimal amount of overdispersion on top of the negative probe count model
- nuclei counts need to be provided in advance and are assumed to be subjected to measurement noise with a CV of 0.1
- non-specific binding does not scale with total counts, but instead scales only with total real counts (after accounting for measurement noise)
- gene expression mean and variance scale with nuclei counts

"""

import sys, ast, os
import time
import itertools
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import theano.tensor as tt
import pymc3 as pm
import pickle
import theano
import string

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from functools import wraps
from adjustText import adjust_text
import matplotlib.cm as cm

from pymc3.distributions.dist_math import bound, logpow, factln, binomln
#from custom_distributions import NegativeBinomial_Adapted

from countcorrect.pymc3_model import Pymc3Model

# defining the model itself
class ProbeCounts_GeneralModel(Pymc3Model):
    r"""NanostringWTA Generative Model:
    :param X_data: Numpy array of gene probe counts (cols) in ROIs (rows)
    :param Y_data: Numpy array of negative probe counts (cols) in ROIs (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    
    This is a more advanced version of the model that includes:

    - overdispersion on top of the gene expression model
    - a minimal amount of overdispersion on top of the negative probe count model
    - nuclei counts need to be provided in advance and are assumed to be subjected to measurement noise with a CV of 0.05
    - non-specific binding does not scale with total counts, but instead scales only with total real counts (after accounting for measurement noise)
    - gene expression mean and variance scale with nuclei counts
    - extra gene level parameter that describes the expected expression level for each gene (e

    """

    def __init__(
        self,
        X_data: np.ndarray,
        Y_data: np.ndarray,
        nuclei: np.ndarray,
        data_type: str = 'float32',
        n_iter = 100000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None, fact_names=None, sample_id=None,
        n_factors = 30,
        v_mu_b_n_hyper = 0.05, # Confidence in the prior distribution for non-specific binding that is automatically set based on negative probe    
                      # data. v_b_n denotes the ratio of standard deviation to mean in the prior distribution. So the default value of 0.05 
                      # implies high confidence that the negative probe counts are representative of non-specific binding in the gene 
                      # probes.
        mu_w_hyp = 0.05, # Hyper prior mean of the weights for each factor in non-negative matrix factorization
        sigma_w_hyp  = 0.1, # Hyper prior standard deviation of weights for each factor in non-negative matrix factorization. A 
                                 # standard deviation higher than the mean implies that most rois are dominated by a few factors, rather 
                                 # than a mixture of many factors.
        v_mu_w_hyp = 0.4,   # Hyper prior standard deviation for the mean of the weights in non-negative matrix factorization, 
                                 # expressed as ratio of the mean. So with a default value at 0.4 the mean can easily go to 0, so that
                                 # factors can be switched off if 'not needed' (Automatic Relevance Detection).
        v_sigma_w_hyp = 0.5, # Hyper prior standard deviation for the standard deviation of the weights in non-negative matrix 
                                 # factorization, expressed as ratio of the mean. So a value of 0.5 denotes relativelly high uncertainty.
        mu_l_v_g = 0.01,
        v_mu_e_g = 0.25
    ):
        
        ############# Initialise parameters ################
        super().__init__(X_data, 0,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)
        self.Y_data = Y_data
        self.y_data = theano.shared(Y_data.astype(self.data_type))
        self.n_rois = Y_data.shape[0]
        self.n_factors = n_factors
        self.n_npro = Y_data.shape[1]
        self.genes = var_names
        self.nuclei_mu = nuclei
        self.l_r = np.array([np.sum(X_data[i,:]) for i in range(self.n_rois)]).reshape(self.n_rois,1)*10**(-5)
        self.v_mu_b_n_hyper = v_mu_b_n_hyper
        self.mu_w_hyp = mu_w_hyp 
        self.sigma_w_hyp  = sigma_w_hyp
        self.v_mu_w_hyp = v_mu_w_hyp
        self.v_sigma_w_hyp = v_sigma_w_hyp
        self.alpha_g = 0.1
        self.v_mu_e_g = v_mu_e_g
        self.mu_l_v_g = mu_l_v_g
        
        if not obs_names:
            self.sample_names = [str(i) for i in range(np.shape(self.X_data)[0])]
        else:
            self.sample_names = obs_names
        
        ############# Define the model ################
        self.model = pm.Model()
        with self.model:

            ### Expected number of nuclei per roi (maybe aadd noise to this later) ###
            
            self.nuclei_r = theano.shared(self.nuclei_mu.reshape(1,self.n_rois).astype(self.data_type))
            
            ### Non-specific binding probability/ Negative Probe Binding Probability Prior Distribution Parameters ###
            
            # Automatically set based on the negative probe counts:
            self.mu_b_n_hyper = np.array((np.mean(np.mean(self.Y_data / self.l_r , axis = 0)), np.var(np.mean(self.Y_data / self.l_r , axis = 0))))
            self.b_n_hyper = pm.Gamma('b_n_hyper', mu = self.mu_b_n_hyper,
                                      sigma = self.mu_b_n_hyper * self.v_mu_b_n_hyper,
                                      shape = 2) + 10**(-9)
            
            ### Gene probe counts model ###
            
            # Background for gene probes (drawn from the same distribution as negative probes):
            self.b_g = pm.Gamma('b_g', mu = self.b_n_hyper[0], sigma = self.b_n_hyper[1], shape = (1,self.n_genes))
            self.B_rg = pm.Deterministic('B_rg', self.l_r*self.b_g)
            
            # Non-negative factors:            
            self.h_gf = pm.Dirichlet('h_gf', a=np.ones(self.n_factors) * self.alpha_g,
                                     shape=(self.n_genes, self.n_factors))
            self.w_hyp = pm.Gamma('w_hyp', mu = np.array((self.mu_w_hyp, self.sigma_w_hyp)),
                                  sigma = np.array((self.mu_w_hyp * self.v_mu_w_hyp, self.sigma_w_hyp * self.v_sigma_w_hyp)),
                                  shape=(self.n_factors,2)) + 10**(-9)
            self.w_rf = pm.Gamma('w_rf', mu=self.w_hyp[:,0], sigma=self.w_hyp[:,1], shape=(self.n_rois, self.n_factors))
            self.A_rg_norm =  pm.math.dot(self.w_rf, self.h_gf.T)
            
            # Define expression level e_g to match mean of data:
            self.mu_e_g = np.mean((self.X_data - np.mean(self.Y_data, axis = 1).reshape(np.shape(Y_data)[0], 1)).clip(min = 0)/self.mu_w_hyp/self.nuclei_mu.reshape(len(self.nuclei_mu),1), axis = 0) + 10**(-9)
            self.e_g = pm.Gamma('e_g', mu = self.mu_e_g, sigma = self.v_mu_e_g * self.mu_e_g, shape = (1,self.n_genes))
            
            # Non-negative matrix factorization times expression level and nuclei counts defines gene expression:
            self.A_rg =  pm.Deterministic('A_rg', tt.extra_ops.repeat(self.nuclei_r.T, self.n_genes, axis=1) * self.A_rg_norm*self.e_g)
            
            # Gene counts expected value is sum of real expression and background:
            self.x_rg = self.A_rg + self.B_rg
            
            # Gene Counts Overdispersion:
            normalized_real_counts = np.mean((self.X_data - np.mean(self.Y_data, axis = 1).reshape(np.shape(self.Y_data)[0],1)).clip(min = 0)/self.nuclei_mu.reshape(len(self.nuclei_mu),1), axis = 0)
            self.l_v_g = pm.Gamma('l_v_g', mu = self.mu_l_v_g, sigma = self.mu_l_v_g, shape=1)
            self.mu_v_g = normalized_real_counts * self.l_v_g + 10**(-9)
            self.v_g = pm.Gamma('v_g', mu = self.mu_v_g, sigma = self.mu_v_g*0.01, shape=(1,self.n_genes))
            # (Calculate alpha instead of v_g to match pymc3 parameterization of negative binomial distribution)
            self.alpha_x = self.x_rg**2/(tt.extra_ops.repeat(self.nuclei_r, self.n_genes, axis=0).T**2*tt.extra_ops.repeat(self.v_g, self.n_rois, axis=0)+ 0.001)
            
            ### Negative Probe Count Model ##
            self.b_n = pm.Gamma('b_n', mu = self.b_n_hyper[0], sigma = self.b_n_hyper[1], shape = (1,self.n_npro))
            self.y_rn = self.b_n*self.l_r
            
            # Negative Probe Counts Overdispersion (basically none)
            self.v_n = pm.Gamma('v_n', mu = 10**(-9), sigma = 10**(-9), shape=(1,self.n_npro)) 
            self.alpha_y = self.y_rn**2/(tt.extra_ops.repeat(self.nuclei_r, self.n_npro, axis=0).T**2*tt.extra_ops.repeat(self.v_n, self.n_rois, axis=0)+ 0.001)
            
            ### Combine gene probes and negative probes:
            
            # Mean, alpha and standard deviation for Negative Binomial Distribution
            self.mu_biol = tt.concatenate([self.y_rn, self.x_rg], axis = 1) + 0.001
            self.alpha_biol = tt.concatenate([self.alpha_y, self.alpha_x], axis = 1) + 0.001
            
            # Calculate NB log probability density
            self.data_target = pm.NegativeBinomial('data_target', mu= self.mu_biol,
                                                   alpha= self.alpha_biol,
                                                   observed=tt.concatenate([self.y_data, self.x_data], axis = 1))
    
    def compute_X_corrected(self):
        r""" Save expected value of negative probe poisson mean and negative probe level"""
        # compute poisson mean of unnormalized gene probe counts:
        self.X_latent_mean = np.round(self.samples['post_sample_means']['A_rg'])
        self.X_latent_q05 = np.round(self.samples['post_sample_q05']['A_rg'])
        self.X_latent_q95 = np.round(self.samples['post_sample_q95']['A_rg'])  
        self.X_latent_q01 = np.round(self.samples['post_sample_q01']['A_rg'])
        self.X_latent_q99 = np.round(self.samples['post_sample_q99']['A_rg'])
        
        self.X_corrected_mean = np.round(np.clip(self.X_data - self.samples['post_sample_means']['B_rg'], a_min = 0, a_max = None))
        self.X_corrected_q05 = np.round(np.clip(self.X_data - self.samples['post_sample_q05']['B_rg'], a_min = 0, a_max = None))
        self.X_corrected_q95 = np.round(np.clip(self.X_data - self.samples['post_sample_q95']['B_rg'], a_min = 0, a_max = None))  
        self.X_corrected_q01 = np.round(np.clip(self.X_data - self.samples['post_sample_q01']['B_rg'], a_min = 0, a_max = None))
        self.X_corrected_q99 = np.round(np.clip(self.X_data - self.samples['post_sample_q99']['B_rg'], a_min = 0, a_max = None))
        
        # also compute naive correction:
        self.X_naive = np.round(np.clip(self.X_data - np.mean(self.Y_data, axis = 1).reshape(np.shape(self.Y_data)[0],1), a_min = 0, a_max = None))
        
    def compute_X_detected(self):
        r"""Compute whether a gene is expressed or not based on whether it has at least 1 count in the X_corrected_q01 matrix"""
        self.X_detected = 1*np.asarray([self.X_corrected_q01[i,:] >= 2 for i in range(np.shape(self.X_corrected_q01)[0])])
        
    def compute_logFC(self, groupA = 3, groupB = 6, n_samples = 1000, correction = 1, normalization = True):
        r"""Compute log-fold change of genes between two ROIs or two groups of ROIs, as well as associated
        standard deviation and 0.05 and 0.95 percentile"""
        
        groupA = np.squeeze(groupA)
        groupB = np.squeeze(groupB)

        if correction == 1:
            X_corrected_post_sample = (self.X_data - self.mean_field['init_1'].sample_node(self.B_rg, size = n_samples).eval()).clip(min = 0)
        elif correction == 2:
            X_corrected_post_sample = self.mean_field['init_1'].sample_node(self.A_rg, size = n_samples).eval()
            
        self.logFC = pd.DataFrame(index = self.genes, columns = ('groupA', 'groupB', 'mean', 'sd', 'q05', 'q95'))

        if normalization:
            total_counts = np.sum(X_corrected_post_sample, axis = 2)
            total_counts[total_counts == 0] = 10**(-9)
            X_corrected_post_sample = X_corrected_post_sample/total_counts.reshape(np.shape(total_counts)[0],np.shape(total_counts)[1],1)*10**6

        if sum(np.shape(groupA)) < 2:
            groupA_value = np.log2(X_corrected_post_sample[:,groupA,:])
            self.logFC['groupA'] = self.sample_names[groupA]
        else:
            groupA_value = np.log2(np.mean(X_corrected_post_sample[:,groupA,:], axis = 1))
            self.logFC['groupA'] = ', '.join([self.sample_names[i] for i in groupA])
        if sum(np.shape(groupB)) < 2:
            groupB_value = np.log2(X_corrected_post_sample[:,groupB,:])
            self.logFC['groupB'] = self.sample_names[groupB]
        else:
            groupB_value = np.log2(np.mean(X_corrected_post_sample[:,groupB,:], axis = 1))
            self.logFC['groupB'] = ', '.join([self.sample_names[i] for i in groupB])

        self.logFC_sample =  groupA_value - groupB_value
        self.logFC['mean'] = self.logFC_sample.mean(axis=0)
        self.logFC['sd'] = self.logFC_sample.std(axis=0) 
        self.logFC['q05'] = np.quantile(self.logFC_sample, 0.05, axis=0)
        self.logFC['q95'] = np.quantile(self.logFC_sample, 0.95, axis=0)  
        
    def compute_FDR(self, logFC_threshold = 1):
        r"""Compute probability that logFC is above a certain threshold 
        and also include FDR for each probability level.
        :logFC_threshold: logFC threshold above which we define a discovery"""
        self.logFC['threshold'] = logFC_threshold
        self.logFC['probability'] = np.sum(np.abs(self.logFC_sample) > logFC_threshold, axis = 0)/np.shape(self.logFC_sample)[0]
        probability = self.logFC['probability']
        self.logFC['FDR'] = np.array([sum(1-probability[probability >= p])/sum(probability >= p) for p in probability])
        self.logFC = self.logFC.sort_values('FDR')
    
    def plot_volcano(self, genesOfInterest = None, n_max_genes = 1, alpha = 0.25, FDR_cutoff = 0.05,
                     height = 10, width = 10):
        r""" Make a volcano plot of the differential expression analysis.
        :genesOfInterest: numpy array of genes to annotate in the plot
        :n_max_genes: number of genes to automatically annotate at the extreme ends of the plot,
        i.e. the most differentially expressed genes
        :alpha: transparency of dots 
        :FDR_cutoff: what false discovery rate to use
        :height: height of figure
        :width: width of figure
        """
        
        # Set figure parameters:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        plt.figure(figsize=(width,height))
        colours = np.repeat('grey', self.n_genes)
        colours[[self.logFC['mean'][i] > 0 and self.logFC['FDR'][i] < FDR_cutoff for i in range(len(self.logFC['mean']))]] = 'blue'
        colours[[self.logFC['mean'][i] < 0 and self.logFC['FDR'][i] < FDR_cutoff for i in range(len(self.logFC['mean']))]] = 'red'
        plt.scatter(self.logFC['mean'], -np.log10(1-self.logFC['probability']+0.0001), c=colours, alpha = 0.1)
        #plt.hlines(np.amin(self.logFC['probability'][self.logFC['FDR'] < FDR_cutoff]), np.amin(self.logFC['mean']),
        #           np.amax(self.logFC['mean']), linestyles = 'dashed')
        #plt.text(np.amin(self.logFC['mean']),np.amin(self.logFC['probability'][self.logFC['FDR'] < FDR_cutoff]) + 0.01,
        #         'FDR < ' + str(FDR_cutoff))
        plt.xlabel('Log2FC')
        plt.ylim(0,3.5)
        plt.ylabel('-log10(P(Log2FC > ' + str(self.logFC['threshold'][0]) + '))')
        
        if n_max_genes > 0:
            
            maxGenes = np.array((self.logFC.index[np.argmax(self.logFC['mean'])],
                                 self.logFC.index[np.argmin(self.logFC['mean'])]))
        if genesOfInterest is None:
            
            genesOfInterest = maxGenes
        else:
            genesOfInterest = np.concatenate((genesOfInterest, maxGenes))
        
        if genesOfInterest is not None:
        
            geneIndex_to_annotate = np.squeeze([np.where(self.logFC.index == genesOfInterest[i])
                                                for i in range(len(genesOfInterest))])
            
            ts = []    
            for i,j in enumerate(geneIndex_to_annotate):
                ts.append(plt.text(self.logFC['mean'][j], -np.log10(1-self.logFC['probability'][j]), genesOfInterest[i]))
            adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 2.5,
                       force_points = 2.5, force_objects = 2.5)
        
        plt.show()   
        
    def plot_factor_weights(self, order = None):
        r""" Plot the factor weights in a heatmap, with the rows (factors) cluster and the samples in the order provided
        :genesOfInterest: numpy array of order (indexes) to put the samples in the heatmap, 
        if None then no reordering will take place
        """ 
        if order is None:
            order = np.arrange(1,len())
        w = self.samples['post_sample_means']['w'][order,:]
        sns.clustermap(w.T, col_cluster=False)        
        
    def plot_negativeProbes_vs_geneBackground(self, samples, n_bins = 100, n_samples = 1000):
        r""" Plot histogram of the negative probe counts and the poisterior poisson mean of background counts
        across all genes
        :samples: numpy array with sequence of samples for which to do this plot
        :n_bins: number of bins for the geneBackground histogram
        :n_samples: number of samples to use for approximating posterior distribution of gene background
        """
        post_sample_b_g = self.mean_field['init_1'].sample_node(self.b_g, size = n_samples).eval()
        
        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle('Negative Probe Counts and Posterior Poisson Mean of all Background Counts')
        
        for i in range(len(samples)):
            ax[0].hist(self.Y_data[:,samples[i]], bins = 10, alpha = 0.75)
        
        ax[0].set_ylabel('Number of Probes')
        
        for i in range(len(samples)):        
            ax[1].hist(np.squeeze(post_sample_b_g[:,:,:]*self.l_r[samples[i]]).flatten(),
                       bins = n_bins, density = True, label = 'sample ' + str(i), alpha = 0.75)

        ax[1].set_xlabel('Counts')
        ax[1].set_ylabel('Probability Density')
        ax[1].set_xscale('log')
        ax[1].legend()
        plt.show()                            
                                              
    def plot_single_geneCounts_and_poissonMean(self, gene, samples, n_samples = 1000):
        r""" Plot a scatter plot of gene counts and a histogram of predicted poisson mean
        :gene: which example gene to plot
        :samples: numpy array with sequence of samples for which to do this plot, maximum is 6
        :n_samples: how many samples to take to approximate posterior
        """
        
        post_sample_A_rg = self.mean_field['init_1'].sample_node(self.A_rg, size = n_samples).eval()
        post_sample_b_g = self.mean_field['init_1'].sample_node(self.b_g, size = n_samples).eval()
        
        if len(samples) > 5:
            print('Maximum Number of Samples is 6')
        colourPalette = c('blue', 'green', 'red', 'yellow', 'black', 'grey')
        
        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle('Raw Counts and Posterior Poisson Mean of Gene Counts (' + gene + ')')
        ax[0].scatter(X_data[samples, self.genes == gene], ('Sample1','Sample2'),  40,
                      c = [colourPalette[i] for i in range(len(samples))])
        
        for i in range(len(samples)):
            ax[1].hist(np.squeeze(post_sample_A_rg[:,i,self.genes == gene]*self.l_r[i]),
                       bins = n_bins, density = True, label = 'sample 1', color = 'blue')
        
        ax[1].set_xlabel('Counts')
        ax[1].set_ylabel('Probability Density')
        ax[1].legend()

        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle('Negative Probe Counts and Posterior Poisson Mean of Background Counts (' + gene + ')')
        
        for i in range(len(samples)):
            ax[0].hist(negProbes_subset[:,samples[i]], color = 'blue', bins = 100, alpha = 0.5)        
        
        ax[0].set_ylabel('Number of Probes')
        
        for i in range(len(samples)):
        
            ax[1].hist(np.squeeze(post_sample_b_g[:,:, self.genes == gene]*self.l_r[samples[i]]), bins = 10,
                       density = True, label = 'sample ' + str(i), color = 'blue', alpha = 0.75)

        ax[1].set_xlabel('Counts')
        ax[1].set_ylabel('Probability Density')
        ax[1].set_xscale('log')
        ax[1].legend()
    
    
    def plot_X_corrected_overview1(self, genesOfInterest, cmap = 'cool', saveFig = None):
        
        r""" Plots a scatter plot of removed counts vs. total counts for each gene probe, colour by fraction removed.
        """

        total_counts_gene = np.sum(self.X_data, axis = 0)
        removed_counts_ISC_gene = np.sum(self.X_data, axis = 0) -  np.sum(self.X_corrected_mean, axis = 0)
        fraction_removed = removed_counts_ISC_gene/total_counts_gene
        
        subset = fraction_removed > 0
        
        removed_counts_ISC_gene = removed_counts_ISC_gene[subset]
        total_counts_gene = total_counts_gene[subset]
        fraction_removed = fraction_removed[subset]
        
        geneIndex_to_annotate = np.squeeze([np.where(self.var_names[subset] == genesOfInterest[i])
                                        for i in range(len(genesOfInterest))])
        
        SMALL_SIZE = 26
        MEDIUM_SIZE = 26
        BIGGER_SIZE = 26
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig, ax = plt.subplots(1,1, figsize=(15,10))

        mesh = ax.scatter(total_counts_gene,  removed_counts_ISC_gene, alpha = 0.5, c = fraction_removed, label = 'ISC Model',
           s=10, cmap = cmap)
        ax.legend()
        ax.set_xlabel('Total counts for gene')
        ax.set_ylabel('Total counts removed from gene')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_ylim(10, 10**7)
        ax=plt.gca()
        cbar = fig.colorbar(mesh)
        cbar.set_label('Fraction Removed')

        ts = []    
        for i,j in enumerate(geneIndex_to_annotate):
            ts.append(plt.text(total_counts_gene[j],  removed_counts_ISC_gene[j], genesOfInterest[i]))
        adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 2.5,
               force_points = 2.5, force_objects = 2.5)
        
        if saveFig:
            plt.savefig(saveFig)
        
        plt.show()
    
    def plot_X_corrected_overview2(self):
        
        r""" Plots various plots showing removed vs. total counts, compared to a naive model that removes 
        the mean number of negative probes counts frome each gene.
        """
        
        SMALL_SIZE = 16
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        total_counts = np.sum(self.X_data, axis = 1)
        removed_counts_ISC = np.sum(self.X_data, axis = 1) - np.sum(self.X_corrected_mean, axis = 1)
        removed_counts_Naive = np.mean(self.Y_data, axis = 1)*np.shape(self.X_data)[1]
        
        total_counts_gene = np.sum(self.X_data, axis = 0)
        removed_counts_ISC_gene = np.sum(self.X_data, axis = 0) -  np.sum(self.X_corrected_mean, axis = 0)
        removed_counts_Naive_gene = np.repeat(np.sum(np.mean(self.Y_data, axis = 1)),len(total_counts_gene))

        fig, ax = plt.subplots(2,2, figsize=(10,10))

        ax[0,0].scatter(removed_counts_Naive, total_counts, alpha = 0.5, color = 'blue', label = 'Naive Model')
        ax[0,0].scatter(removed_counts_ISC, total_counts, alpha = 0.5, color = 'red', label = 'ISC Model')
        ax[0,0].legend()
        ax[0,0].set_ylabel('Total counts in ROI/AOI')
        ax[0,0].set_xlabel('Total counts removed from ROI/AOI')
        ax[0,0].set_xscale('log')
        ax[0,0].set_yscale('log')
        ax[0,0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax[0,0].transAxes, 
            size=20, weight='bold')

        ax[1,0].hist(removed_counts_Naive/total_counts, label = 'Naive Model',
                 color = 'blue')
        ax[1,0].hist(removed_counts_ISC/total_counts, label = 'ISC model', color = 'red', bins = 4)
        ax[1,0].set_xlabel('Fraction of counts removed from ROI/AOI')
        ax[1,0].set_ylabel('Occurences')
        ax[1,0].legend()
        ax[1,0].set_xlim(0,1)
        ax[1,0].text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax[1,0].transAxes, 
            size=20, weight='bold')

        ax[0,1].scatter(removed_counts_ISC_gene, total_counts_gene,  alpha = 0.25, color = 'red', label = 'ISC Model', s=5)
        ax[0,1].scatter(removed_counts_Naive_gene, total_counts_gene,  alpha = 0.25, color = 'blue', label = 'Naive Model', s=5)
        ax[0,1].legend()
        ax[0,1].set_ylabel('Total counts for gene')
        ax[0,1].set_xlabel('Total counts removed from gene')
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].set_xlim(10**2, 10**6)
        ax[0,1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax[0,1].transAxes, 
            size=20, weight='bold')

        ax[1,1].hist(np.clip(removed_counts_Naive_gene/total_counts_gene,0,1), label = 'Naive Model',
                 alpha = 0.5, color = 'blue', bins = 10)
        ax[1,1].hist(removed_counts_ISC_gene/total_counts_gene, label = 'ISC model', alpha = 0.5, color = 'red', bins = 20)
        ax[1,1].set_xlabel('Fraction of counts removed from gene')
        ax[1,1].set_xlim(0,1)
        ax[1,1].set_ylabel('Occurences')
        ax[1,1].legend()
        ax[1,1].text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax[1,1].transAxes, 
            size=20, weight='bold')
        plt.tight_layout()
        
    def plot_X_corrected_overview3(self, saveFig = None):
        
        r""" Plots various plots showing removed vs. total counts, compared to a naive model that removes 
        the mean number of negative probes counts frome each gene.
        """
                
        SMALL_SIZE = 20
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=20)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        total_counts = np.sum(self.X_data, axis = 1)
        removed_counts_ISC = np.sum(self.X_data, axis = 1) - np.sum(self.X_corrected_mean, axis = 1)
        removed_counts_Naive = np.mean(self.Y_data, axis = 1)*np.shape(self.X_data)[1]
        
        total_counts_gene = np.sum(self.X_data, axis = 0)
        removed_counts_ISC_gene = np.sum(self.X_data, axis = 0) -  np.sum(self.X_corrected_mean, axis = 0)
        removed_counts_Naive_gene = np.repeat(np.sum(np.mean(self.Y_data, axis = 1)),len(total_counts_gene))

        fig, ax = plt.subplots(1,1, figsize=(7,5))

        ax.hist(removed_counts_ISC_gene/total_counts_gene, label = 'ISC model', alpha = 0.5, color = 'red')
        ax.set_xlabel('Fraction of counts removed from gene')
        ax.set_xlim(0,1)
        ax.set_ylabel('Occurences')
        ax.legend()

        plt.tight_layout()
        
        if saveFig:
            plt.savefig(saveFig)
            

    def plot_X_corrected_overview6(self, genesOfInterest, cmap = 'cool', saveFig = None):

        r""" Plots a scatter plot of removed counts vs. total counts for each gene probe, colour by fraction removed. 
        In addition, includes a line for the naive model. This version shows mean instead of total counts.
        This version shows CPM instead of raw counts"""
        
        total_counts = np.sum(self.X_data, axis = 1)
        cpm_counts_gene =  np.mean(np.asarray([self.X_data[i,:]/total_counts[i] for i in range(len(total_counts))])*10**6, axis = 0)
        total_counts_gene = np.mean(self.X_data, axis = 0)
        removed_counts_ISC_gene = np.mean(self.X_data, axis = 0) -  np.mean(self.X_corrected_mean, axis = 0)
        fraction_removed = removed_counts_ISC_gene/total_counts_gene
        removed_counts_Naive = np.repeat(np.mean(np.mean(self.Y_data, axis = 1)), np.shape(self.X_data)[1])
        down2sd = removed_counts_Naive - np.repeat(np.mean(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
        up2sd = removed_counts_Naive + np.repeat(np.mean(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
        
        subset = fraction_removed > 0

        removed_counts_ISC_gene = removed_counts_ISC_gene[subset]
        total_counts_gene = total_counts_gene[subset]
        fraction_removed = fraction_removed[subset]
        removed_counts_Naive = removed_counts_Naive[subset]
        cpm_counts_gene = cpm_counts_gene[subset]

        geneIndex_to_annotate = np.squeeze([np.where(self.var_names[subset] == genesOfInterest[i])
                                        for i in range(len(genesOfInterest))])
        SMALL_SIZE = 26
        MEDIUM_SIZE = 26
        BIGGER_SIZE = 26
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig, ax = plt.subplots(1,1, figsize=(20,13.5))
        
        #ax.plot(total_counts_gene, removed_counts_Naive, c = 'black', label = 'Negative Probe Mean')
        ax.axvline(x=removed_counts_Naive[0], c = 'black', label = 'Negative Probe Mean', linestyle = '-')
        ax.axhline(y=removed_counts_Naive[0], c = 'black', linestyle = '-')
        ax.axvline(x=up2sd[0], c = 'grey', label = '+- 2 Standard Deviations', linestyle = '--')
        ax.axhline(y=up2sd[0], c = 'grey', linestyle = '--')
        ax.axvline(x=down2sd[0], c = 'grey', linestyle = '--')
        ax.axhline(y=down2sd[0], c = 'grey', linestyle = '--')
        mesh = ax.scatter(cpm_counts_gene,  removed_counts_ISC_gene, alpha = 0.5, c = fraction_removed, label = 'CountCorrect',
           s=10, cmap = cmap)
        ax.legend()
        ax.set_xlabel('Mean counts across ROIs')
        ax.set_ylabel('Mean estimated background counts')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.15, 10**4)
        ax.set_facecolor('white')
        ax=plt.gca()
        cbar = fig.colorbar(mesh)
        cbar.set_label('Fraction Removed')
            
        ts = []    
        for i,j in enumerate(geneIndex_to_annotate):
            ts.append(plt.text(total_counts_gene[j],  removed_counts_ISC_gene[j], genesOfInterest[i]))
        adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 2.5,
               force_points = 2.5, force_objects = 2.5)

        if saveFig:
            plt.savefig(saveFig)

        plt.show()            
            
    def plot_X_corrected_overview5(self, genesOfInterest, cmap = 'cool', saveFig = None, correction = 1):

        r""" Plots a scatter plot of removed counts vs. total counts for each gene probe, colour by fraction removed. 
        In addition, includes a line for the naive model. This version shows mean instead of total counts """
        
        total_counts_gene = np.mean(self.X_data, axis = 0)
        
        if correction == 1:
        
            removed_counts_ISC_gene = np.mean(self.X_data, axis = 0) -  np.mean(self.X_corrected_mean, axis = 0)
            
        elif correction == 2:
            
            removed_counts_ISC_gene = np.mean(self.X_data, axis = 0) -  np.mean(self.X_latent_mean, axis = 0)
            
        fraction_removed = removed_counts_ISC_gene/total_counts_gene
        removed_counts_Naive = np.repeat(np.mean(np.mean(self.Y_data, axis = 1)), np.shape(self.X_data)[1])
        down2sd = removed_counts_Naive - np.repeat(np.mean(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
        up2sd = removed_counts_Naive + np.repeat(np.mean(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
        
        subset = fraction_removed > 0

        removed_counts_ISC_gene = removed_counts_ISC_gene[subset]
        total_counts_gene = total_counts_gene[subset]
        fraction_removed = fraction_removed[subset]

        geneIndex_to_annotate = np.squeeze([np.where(self.var_names[subset] == genesOfInterest[i])
                                        for i in range(len(genesOfInterest))])
        SMALL_SIZE = 26
        MEDIUM_SIZE = 26
        BIGGER_SIZE = 26
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig, ax = plt.subplots(1,1, figsize=(20,13.5))
        
        #ax.plot(total_counts_gene, removed_counts_Naive, c = 'black', label = 'Negative Probe Mean')
        ax.axvline(x=removed_counts_Naive[0], c = 'black', label = 'Negative Probe Mean', linestyle = '-')
        ax.axhline(y=removed_counts_Naive[0], c = 'black', linestyle = '-')
        ax.axvline(x=up2sd[0], c = 'grey', label = '+- 2 Standard Deviations', linestyle = '--')
        ax.axhline(y=up2sd[0], c = 'grey', linestyle = '--')
        ax.axvline(x=down2sd[0], c = 'grey', linestyle = '--')
        ax.axhline(y=down2sd[0], c = 'grey', linestyle = '--')
        mesh = ax.scatter(total_counts_gene,  removed_counts_ISC_gene, alpha = 0.5, c = fraction_removed, label = 'CountCorrect',
           s=10, cmap = cmap)
        ax.legend()
        ax.set_xlabel('Mean counts across ROIs')
        ax.set_ylabel('Mean estimated background counts')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.01, 10**4)
        ax.set_facecolor('white')
        ax=plt.gca()
        cbar = fig.colorbar(mesh)
        cbar.set_label('Fraction Removed')
            
        ts = []    
        for i,j in enumerate(geneIndex_to_annotate):
            ts.append(plt.text(total_counts_gene[j],  removed_counts_ISC_gene[j], genesOfInterest[i]))
        adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 2.5,
               force_points = 2.5, force_objects = 2.5)

        if saveFig:
            plt.savefig(saveFig)

        plt.show()
        
        def plot_X_corrected_overview4(self, genesOfInterest, cmap = 'cool', saveFig = None):

            r""" Plots a scatter plot of removed counts vs. total counts for each gene probe, colour by fraction removed. 
            In addition, includes a line for the naive model """

            total_counts_gene = np.sum(self.X_data, axis = 0)
            removed_counts_ISC_gene = np.sum(self.X_data, axis = 0) -  np.sum(self.X_corrected_mean, axis = 0)
            fraction_removed = removed_counts_ISC_gene/total_counts_gene
            removed_counts_Naive = np.repeat(np.sum(np.mean(self.Y_data, axis = 1)), np.shape(self.X_data)[1])
            down2sd = removed_counts_Naive - np.repeat(np.sum(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
            up2sd = removed_counts_Naive + np.repeat(np.sum(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])

            subset = fraction_removed > 0

            removed_counts_ISC_gene = removed_counts_ISC_gene[subset]
            total_counts_gene = total_counts_gene[subset]
            fraction_removed = fraction_removed[subset]
            removed_counts_Naive = removed_counts_Naive[subset]

            geneIndex_to_annotate = np.squeeze([np.where(self.var_names[subset] == genesOfInterest[i])
                                            for i in range(len(genesOfInterest))])
            SMALL_SIZE = 26
            MEDIUM_SIZE = 26
            BIGGER_SIZE = 26
            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            fig, ax = plt.subplots(1,1, figsize=(20,13.5))

            #ax.plot(total_counts_gene, removed_counts_Naive, c = 'black', label = 'Negative Probe Mean')
            ax.axvline(x=removed_counts_Naive[0], c = 'black', label = 'Negative Probe Mean', linestyle = '-')
            ax.axhline(y=removed_counts_Naive[0], c = 'black', linestyle = '-')
            ax.axvline(x=up2sd[0], c = 'grey', label = '+- 2 Standard Deviationns', linestyle = '--')
            ax.axhline(y=up2sd[0], c = 'grey', linestyle = '--')
            ax.axvline(x=down2sd[0], c = 'grey', linestyle = '--')
            ax.axhline(y=down2sd[0], c = 'grey', linestyle = '--')
            mesh = ax.scatter(total_counts_gene,  removed_counts_ISC_gene, alpha = 0.5, c = fraction_removed, label = 'CountCorrect',
               s=10, cmap = cmap)
            ax.legend()
            ax.set_xlabel('Total counts for gene')
            ax.set_ylabel('Total counts removed from gene')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(10, 10**7)
            ax.set_facecolor('white')
            ax=plt.gca()
            cbar = fig.colorbar(mesh)
            cbar.set_label('Fraction Removed')

            ts = []    
            for i,j in enumerate(geneIndex_to_annotate):
                ts.append(plt.text(total_counts_gene[j],  removed_counts_ISC_gene[j], genesOfInterest[i]))
            adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 2.5,
                   force_points = 2.5, force_objects = 2.5)

            if saveFig:
                plt.savefig(saveFig)

            plt.show()    
        
    def rank_X_corrected_genes(self):
        
        mean_counts = np.mean(self.X_data, axis = 0)
        total_counts = np.sum(self.X_data, axis = 0)
        removed_counts = np.sum(self.X_data, axis = 0) - np.sum(self.X_corrected_mean, axis = 0)
        fraction_removed_counts = removed_counts/total_counts
        
        ranked_genes_tab = pd.DataFrame(columns = ('Gene', 'Total Counts', 'Removed Counts', 'Fraction Removed Counts (Mean)'))
                                        
        ranked_genes_tab['Gene'] = self.var_names
        ranked_genes_tab['Total Counts'] = [int(total_counts[i]) for i in range(len(total_counts))]
        ranked_genes_tab['Mean Counts'] = [int(mean_counts[i]) for i in range(len(mean_counts))]
        ranked_genes_tab['Removed Counts'] = [int(removed_counts[i]) for i in range(len(removed_counts))] 
        ranked_genes_tab['Fraction Removed Counts (Mean)'] = fraction_removed_counts
        
        return ranked_genes_tab.sort_values('Fraction Removed Counts (Mean)', ascending = False)
    
    def plot_X_corrected_exampleGenes(self, x, order, example_genes):
    
        n_example_genes = len(example_genes)
        fig, ax = plt.subplots(n_example_genes,3, figsize=(8*3,n_example_genes*5))

        total_counts = np.sum(self.X_data, axis = 1) 
        
        total_counts_corrected = np.sum(self.X_corrected_mean, axis = 1) 
        
        for i in range(n_example_genes):
            ax[i,0].scatter(x[order], self.X_corrected_mean[order,self.var_names == example_genes[i]]/total_counts_corrected[order]*10**(6),
                           color = 'red', label = 'ISC-CPM')
            ax[i,0].scatter(x[order], self.X_data[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'blue', label = 'CPM')
            ax[i,0].set_title(example_genes[i])
            ax[i,0].legend()
            plt.tight_layout()
            
            ax[i,1].scatter(x[order], self.X_corrected_mean[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'red', label = 'ISC-CPM')
            ax[i,1].scatter(x[order], self.X_data[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'blue', label = 'CPM')
            ax[i,1].set_title(example_genes[i])
            ax[i,1].legend()
            plt.tight_layout()
            
            ax[i,2].scatter(x[order], self.X_corrected_mean[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'red', label = 'ISC-CPM')
            ax[i,2].scatter(x[order], self.X_data[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'blue', label = 'CPM')
            ax[i,2].set_title(example_genes[i])
            ax[i,2].set_yscale('log')
            #ax[i,0].set_ylim(10**(-5), 10**(-3))
            ax[i,2].legend()
            plt.tight_layout()
            
    def plot_X_corrected_exampleGenes1(self, x, order, example_genes, saveFig, naiveModel = False):
    
        n_example_genes = len(example_genes)

        total_counts = np.sum(self.X_data, axis = 1) 

        for i in range(n_example_genes):

            fig, ax = plt.subplots(1,1, figsize=(12,9))
            
            ax.scatter(x[order], self.X_corrected_mean[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'red', label = 'CountCorrect CPM', alpha = 0.75, s = 250)
            ax.scatter(x[order], self.X_data[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'blue', label = 'Raw CPM', alpha = 0.75, s = 100)
            ax.scatter(x[order], self.X_naive[order,self.var_names == example_genes[i]]/total_counts[order]*10**(6),
                           color = 'yellow', label = 'Raw CPM - Mean Negative Probe Counts', alpha = 0.75, s = 100)
            ax.set_title(example_genes[i])
            ax.legend()
            ax.set_ylim(ymin = 0)
            ax.set_ylabel('CPM')
            ax.set_xlabel('Cortical Depth')
            plt.tight_layout()

            if saveFig:
                plt.savefig(saveFig + '_' + example_genes[i] + '.pdf')

            plt.show()
            
    def plot_X_detected(self, x, order, LoD_cutoff, xlabel = 'Cortical Depth'):
        
        X_detected_LoD = 1*np.asarray([self.X_data[i,:] > LoD_cutoff[i] for i in range(np.shape(self.X_data)[0])])
        
        detected_LoD = np.sum(X_detected_LoD[order,:], axis = 1)
        detected_ISC = np.sum(self.X_detected[order,:], axis = 1)
        
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        
        ax[0].scatter(x[order], detected_ISC,
                       color = 'red', label = 'ISC')
        ax[0].scatter(x[order], detected_LoD,
                       color = 'blue', label = 'LoD')
        ax[0].set_title('Number of detected Genes')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylim(0,19000)
        ax[0].legend()
        
        ax[1].scatter(x[order], np.sum(self.X_data, axis = 1)[order])
        ax[1].set_title('Total Counts')
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylim(0,1.5*10**7)
        plt.tight_layout()
        plt.show()
        
    def plot_prior_sample(self):

        fig, ax = plt.subplots(4,3,figsize=(15,15))

        data_node = 'X_data'
        data_target_name = 'data_target'

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name][:,:,self.n_npro:]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        data_node = np.log10(data_node + 1)
        data_target_name = np.log10(data_target_name + 1)

        ax[0,0].hist2d(data_node.flatten(),
                   data_target_name.flatten(),
                   bins=50, norm=matplotlib.colors.LogNorm())
        ax[0,0].set_xlabel('X_data observed, log10(nUMI)')
        ax[0,0].set_ylabel('X_data prior, log10(nUMI)')
        ax[0,0].set_title('X_data prior vs X_data observed')

        ax[0,1].hist(data_node.flatten(),bins=50)
        ax[0,1].set_xlabel('X_data observed, log10(nUMI)')
        ax[0,1].set_ylabel('Occurences')
        ax[0,1].set_title('X_data observed')

        ax[0,2].hist(data_target_name.flatten(),bins=50)
        ax[0,2].set_xlabel('X_data prior, log10(nUMI)')
        ax[0,2].set_ylabel('Occurences')
        ax[0,2].set_title('X_data prior')

        data_node = 'Y_data'
        data_target_name = 'data_target'

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name][:,:,:self.n_npro]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        data_node = np.log10(data_node + 1)
        data_target_name = np.log10(data_target_name + 1)

        ax[1,0].hist2d(data_node.flatten(),
                   data_target_name.flatten(),
                   bins=50, norm=matplotlib.colors.LogNorm())
        ax[1,0].set_xlabel('Y_data observed, log10(nUMI)')
        ax[1,0].set_ylabel('Y_data prior, log10(nUMI)')
        ax[1,0].set_title('Y_data prior vs Y_data observed')

        ax[1,1].hist(data_node.flatten(),bins=50)
        ax[1,1].set_xlabel('Y_data observed, log10(nUMI)')
        ax[1,1].set_ylabel('Occurences')
        ax[1,1].set_title('Y_data observed')

        ax[1,2].hist(data_target_name.flatten(),bins=50)
        ax[1,2].set_xlabel('Y_data prior, log10(nUMI)')
        ax[1,2].set_ylabel('Occurences')
        ax[1,2].set_title('Y_data prior')

        data_node = 'X_data'
        data_target_name = 'A_rg'

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        data_node = np.log10(data_node + 1)
        data_target_name = np.log10(data_target_name + 1)

        ax[2,0].hist2d(data_node.flatten(),
                   data_target_name.flatten(),
                   bins=50, norm=matplotlib.colors.LogNorm())
        ax[2,0].set_xlabel('X_data observed, log10(nUMI)')
        ax[2,0].set_ylabel('X_corrected prior, log10(nUMI)')
        ax[2,0].set_title('X_corrected prior vs X_data observed')

        ax[2,1].hist(data_node.flatten(),bins=50)
        ax[2,1].set_xlabel('X_data observed, log10(nUMI)')
        ax[2,1].set_ylabel('Occurences')
        ax[2,1].set_title('X_data observed')

        ax[2,2].hist(data_target_name.flatten(),bins=50)
        ax[2,2].set_xlabel('X_corrected prior, log10(nUMI)')
        ax[2,2].set_ylabel('Occurences')
        ax[2,2].set_title('X_corrected prior')
        
        data_node = 'X_data'
        data_target_name = 'B_rg'

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        data_node = np.log10(data_node + 1)
        data_target_name = np.log10(data_target_name + 1)

        ax[3,0].hist2d(data_node.flatten(),
                   data_target_name.flatten(),
                   bins=50, norm=matplotlib.colors.LogNorm())
        ax[3,0].set_xlabel('X_data observed, log10(nUMI)')
        ax[3,0].set_ylabel('Non-specific counts prior, log10(nUMI)')
        ax[3,0].set_title('Non-specific counts prior vs X_data observed')
        plt.tight_layout()

        ax[3,1].hist(data_node.flatten(),bins=50)
        ax[3,1].set_xlabel('X_data observed, log10(nUMI)')
        ax[3,1].set_ylabel('Occurences')
        ax[3,1].set_title('X_data observed')

        ax[3,2].hist(data_target_name.flatten(),bins=50)
        ax[3,2].set_xlabel('Non-specific counts prior, log10(nUMI)')
        ax[3,2].set_ylabel('Occurences')
        ax[3,2].set_title('Non-specific counts prior')
        
        plt.tight_layout()
    
        
    def plot_prior_overview(self):
        
        r""" Plots various plots showing removed vs. total counts, compared to a naive model that removes 
        the mean number of negative probes counts frome each gene.
        """
        
        SMALL_SIZE = 16
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        mean_X_rg = np.mean(self.prior_trace['data_target'][:,:,self.n_npro:], axis = 0)
        mean_A_rg = np.mean(self.prior_trace['A_rg'], axis = 0)
        mean_B_rg = np.mean(self.prior_trace['B_rg'], axis = 0)
        
        self.X_corrected_mean_prior = self.X_data - mean_B_rg
        
        total_counts = np.sum(self.X_data, axis = 1)
        removed_counts_ISC = np.sum(self.X_data, axis = 1) - np.sum(self.X_corrected_mean_prior, axis = 1)
        removed_counts_Naive = np.mean(self.Y_data, axis = 1)*np.shape(self.X_data)[1]
        
        total_counts_gene = np.sum(self.X_data, axis = 0)
        removed_counts_ISC_gene = np.sum(self.X_data, axis = 0) -  np.sum(self.X_corrected_mean_prior, axis = 0)
        removed_counts_Naive_gene = np.repeat(np.sum(np.mean(self.Y_data, axis = 1)),len(total_counts_gene))

        fig, ax = plt.subplots(2,2, figsize=(10,10))

        ax[0,0].scatter(removed_counts_Naive, total_counts, alpha = 0.5, color = 'blue', label = 'Naive Model')
        ax[0,0].scatter(removed_counts_ISC, total_counts, alpha = 0.5, color = 'red', label = 'ISC Model')
        ax[0,0].legend()
        ax[0,0].set_ylabel('Total counts in ROI/AOI')
        ax[0,0].set_xlabel('Total counts removed from ROI/AOI')
        ax[0,0].set_xscale('log')
        ax[0,0].set_yscale('log')
        ax[0,0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax[0,0].transAxes, 
            size=20, weight='bold')

        ax[1,0].hist(removed_counts_Naive/total_counts, label = 'Naive Model',
                 color = 'blue')
        ax[1,0].hist(removed_counts_ISC/total_counts, label = 'ISC model', color = 'red', bins = 4)
        ax[1,0].set_xlabel('Fraction of counts removed from ROI/AOI')
        ax[1,0].set_ylabel('Occurences')
        ax[1,0].legend()
        ax[1,0].set_xlim(0,1)
        ax[1,0].text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax[1,0].transAxes, 
            size=20, weight='bold')

        ax[0,1].scatter(removed_counts_ISC_gene, total_counts_gene,  alpha = 0.25, color = 'red', label = 'ISC Model', s=5)
        ax[0,1].scatter(removed_counts_Naive_gene, total_counts_gene,  alpha = 0.25, color = 'blue', label = 'Naive Model', s=5)
        ax[0,1].legend()
        ax[0,1].set_ylabel('Total counts for gene')
        ax[0,1].set_xlabel('Total counts removed from gene')
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].set_xlim(10**2, 10**6)
        ax[0,1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax[0,1].transAxes, 
            size=20, weight='bold')

        ax[1,1].hist(np.clip(removed_counts_Naive_gene/total_counts_gene,0,1), label = 'Naive Model',
                 alpha = 0.5, color = 'blue', bins = 10)
        ax[1,1].hist(removed_counts_ISC_gene/total_counts_gene, label = 'ISC model', alpha = 0.5, color = 'red', bins = 20)
        ax[1,1].set_xlabel('Fraction of counts removed from gene')
        ax[1,1].set_xlim(0,1)
        ax[1,1].set_ylabel('Occurences')
        ax[1,1].legend()
        ax[1,1].text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax[1,1].transAxes, 
            size=20, weight='bold')
        plt.tight_layout()
        
    def plot_prior_posterior_comparison1(self):
        
        r""" Plots various plots showing removed vs. total counts, compared to a naive model that removes 
        the mean number of negative probes counts frome each gene.
        """
        
        SMALL_SIZE = 16
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        mean_X_rg = np.mean(self.prior_trace['data_target'][:,:,self.n_npro:], axis = 0)
        mean_A_rg = np.mean(self.prior_trace['A_rg'], axis = 0)
        mean_B_rg = np.mean(self.prior_trace['B_rg'], axis = 0)
        
        self.X_corrected_mean = self.X_data - mean_B_rg
        
        total_counts = np.sum(self.X_data, axis = 1)
        removed_counts_ISC = np.sum(self.X_data, axis = 1) - np.sum(self.X_corrected_mean, axis = 1)
        removed_counts_Naive = np.mean(self.Y_data, axis = 1)*np.shape(self.X_data)[1]
        
        total_counts_gene = np.sum(self.X_data, axis = 0)
        removed_counts_ISC_gene = np.sum(self.X_data, axis = 0) -  np.sum(self.X_corrected_mean, axis = 0)
        removed_counts_Naive_gene = np.repeat(np.sum(np.mean(self.Y_data, axis = 1)),len(total_counts_gene))

        fig, ax = plt.subplots(2,2, figsize=(10,10))

        ax[0,0].scatter(removed_counts_Naive, total_counts, alpha = 0.5, color = 'blue', label = 'Naive Model')
        ax[0,0].scatter(removed_counts_ISC, total_counts, alpha = 0.5, color = 'red', label = 'ISC Model')
        ax[0,0].legend()
        ax[0,0].set_ylabel('Total counts in ROI/AOI')
        ax[0,0].set_xlabel('Total counts removed from ROI/AOI')
        ax[0,0].set_xscale('log')
        ax[0,0].set_yscale('log')
        ax[0,0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax[0,0].transAxes, 
            size=20, weight='bold')

        ax[1,0].hist(removed_counts_Naive/total_counts, label = 'Naive Model',
                 color = 'blue')
        ax[1,0].hist(removed_counts_ISC/total_counts, label = 'ISC model', color = 'red', bins = 4)
        ax[1,0].set_xlabel('Fraction of counts removed from ROI/AOI')
        ax[1,0].set_ylabel('Occurences')
        ax[1,0].legend()
        ax[1,0].set_xlim(0,1)
        ax[1,0].text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax[1,0].transAxes, 
            size=20, weight='bold')

        ax[0,1].scatter(removed_counts_ISC_gene, total_counts_gene,  alpha = 0.25, color = 'red', label = 'ISC Model', s=5)
        ax[0,1].scatter(removed_counts_Naive_gene, total_counts_gene,  alpha = 0.25, color = 'blue', label = 'Naive Model', s=5)
        ax[0,1].legend()
        ax[0,1].set_ylabel('Total counts for gene')
        ax[0,1].set_xlabel('Total counts removed from gene')
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].set_xlim(10**2, 10**6)
        ax[0,1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax[0,1].transAxes, 
            size=20, weight='bold')

        ax[1,1].hist(np.clip(removed_counts_Naive_gene/total_counts_gene,0,1), label = 'Naive Model',
                 alpha = 0.5, color = 'blue', bins = 10)
        ax[1,1].hist(removed_counts_ISC_gene/total_counts_gene, label = 'ISC model', alpha = 0.5, color = 'red', bins = 20)
        ax[1,1].set_xlabel('Fraction of counts removed from gene')
        ax[1,1].set_xlim(0,1)
        ax[1,1].set_ylabel('Occurences')
        ax[1,1].legend()
        ax[1,1].text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax[1,1].transAxes, 
            size=20, weight='bold')
        plt.tight_layout()
        
    def plot_prior_posterior_comparison2(self, n_samples, n_genes = 100, alpha1 = 0.002, alpha2 = 0.1, size1 = 100, size2 = 100,
                                        correction = 1, ymin = 10, ymax = 10**7, saveFig = False):

        r""" Plots a scatter plot of removed counts vs. total counts for each gene probe, colour by fraction removed. 
        In addition, includes a line for the naive model """
        
        if correction == 1:
        
            self.X_corrected_prior = (self.X_data - self.prior_trace['B_rg']).clip(0)
            self.X_corrected_posterior = (self.X_data - self.mean_field['init_1'].sample_node(self.B_rg, size = n_samples).eval()).clip(0)
            
        elif correction == 2:
            
            self.X_corrected_prior = self.prior_trace['A_rg']
            self.X_corrected_posterior = self.mean_field['init_1'].sample_node(self.A_rg, size = n_samples).eval()

        total_counts_gene = np.sum(self.X_data, axis = 0)
        removed_counts_ISC_gene_prior = np.sum(self.X_data -  self.X_corrected_prior, axis = 1)
        removed_counts_ISC_gene_posterior = np.sum(self.X_data -  self.X_corrected_posterior, axis = 1)
        fraction_removed_prior = removed_counts_ISC_gene_prior/total_counts_gene
        fraction_removed_posterior = removed_counts_ISC_gene_posterior/total_counts_gene
        removed_counts_Naive = np.repeat(np.sum(np.mean(self.Y_data, axis = 1)), np.shape(self.X_data)[1])
        down2sd = removed_counts_Naive - np.repeat(np.sum(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
        up2sd = removed_counts_Naive + np.repeat(np.sum(2*np.sqrt(np.var(self.Y_data, axis = 1))), np.shape(self.X_data)[1])
        total_counts_gene_mean = total_counts_gene
        total_counts_gene = np.tile(total_counts_gene,(np.shape(removed_counts_ISC_gene_posterior)[0],1))
        removed_counts_ISC_gene_prior_mean = np.mean(removed_counts_ISC_gene_prior, axis = 0)
        removed_counts_ISC_gene_posterior_mean = np.mean(removed_counts_ISC_gene_posterior, axis = 0)
        values = np.random.uniform(low = np.log10(np.min(total_counts_gene_mean)), high = np.log10(np.max(total_counts_gene_mean)),size = n_genes)
        array = np.log10(np.asarray(total_counts_gene_mean))
        idx = np.repeat(0, len(values))
        for i in range(len(values)):
            idx[i] = (np.abs(array - values[i])).argmin()

        SMALL_SIZE = 26
        MEDIUM_SIZE = 26
        BIGGER_SIZE = 26
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig, ax = plt.subplots(2,1, figsize=(27,40))

        ax[0].axvline(x=removed_counts_Naive[0], c = 'black', label = 'Negative Probe Mean', linestyle = '-')
        ax[0].axhline(y=removed_counts_Naive[0], c = 'black', linestyle = '-')
        ax[0].axvline(x=up2sd[0], c = 'grey', label = '+- 2 Standard Deviationns', linestyle = '--')
        ax[0].axhline(y=up2sd[0], c = 'grey', linestyle = '--')
        ax[0].axvline(x=down2sd[0], c = 'grey', linestyle = '--')
        ax[0].axhline(y=down2sd[0], c = 'grey', linestyle = '--')
        ax[0].scatter(total_counts_gene[:,idx].flatten(),  removed_counts_ISC_gene_prior[:,idx].flatten(), alpha = alpha1,
        label = 'CountCorrect Samples',
           s=size1, c = 'grey')
        ax[0].scatter(total_counts_gene_mean[idx],  removed_counts_ISC_gene_prior_mean[idx], alpha = alpha2, label = 'CountCorrect Mean',
           s=size2, c = 'blue')
        ax[0].legend()
        ax[0].set_xlabel('Total counts for gene')
        ax[0].set_ylabel('Total counts removed from gene')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_ylim(ymin, ymax)
        ax[0].set_facecolor('white')
        ax[0].set_title('Prior Distribution')

        ax[1].axvline(x=removed_counts_Naive[0], c = 'black', label = 'Negative Probe Mean', linestyle = '-')
        ax[1].axhline(y=removed_counts_Naive[0], c = 'black', linestyle = '-')
        ax[1].axvline(x=up2sd[0], c = 'grey', label = '+- 2 Standard Deviationns', linestyle = '--')
        ax[1].axhline(y=up2sd[0], c = 'grey', linestyle = '--')
        ax[1].axvline(x=down2sd[0], c = 'grey', linestyle = '--')
        ax[1].axhline(y=down2sd[0], c = 'grey', linestyle = '--')
        ax[1].scatter(total_counts_gene[:,idx].flatten(),  removed_counts_ISC_gene_posterior[:,idx].flatten(), alpha = alpha1,
           label = 'CountCorrect - Samples',
           s=size1, c = 'grey')
        ax[1].scatter(total_counts_gene_mean[idx],  removed_counts_ISC_gene_posterior_mean[idx], alpha = alpha2,
           label = 'CountCorrect - Mean',
           s=size2, c = 'blue')
        ax[1].legend()
        ax[1].set_xlabel('Total counts for gene')
        ax[1].set_ylabel('Total counts removed from gene')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_ylim(ymin, ymax)
        ax[1].set_facecolor('white')
        ax[1].set_title('Posterior Distribution')
        
        if saveFig:
            plt.savefig(saveFig)
        
        plt.show() 

        
        
        
        