### Function for running the CountCorrect algorithm.

# -*- coding: utf-8 -*-
r"""Run CountCorrect model to remove backgorund noise from Nanostring WTA data"""

import matplotlib.pyplot as plt
from countcorrect.ProbeCounts__GeneralModel import ProbeCounts_GeneralModel
from countcorrect.ProbeCounts__GeneralModel_V2 import ProbeCounts_GeneralModel_V2
import numpy as np

def run_countcorrect(counts_geneProbes, counts_negativeProbes, counts_nuclei,
                     n_factors = 30, slide_id = None,
                     total_iterations = 20000,
                     learning_rate = 0.01,
                     posterior_samples = 10,
                     verbose = True,
                     naive = False,
                     return_rawCounts = True,
                     return_normalizedCounts = True,
                     return_model = False):
    
    if type(counts_geneProbes).__module__ != np.__name__:
        counts_geneProbes = np.array(counts_geneProbes)
    if type(counts_negativeProbes).__module__ != np.__name__:
        counts_negativeProbes = np.array(counts_negativeProbes)
    if type(counts_nuclei).__module__ != np.__name__:
        counts_nuclei = np.array(counts_nuclei)
    if type(slide_id).__module__ != np.__name__:
        slide_id = np.array(slide_id)   
    counts_nuclei = counts_nuclei.flatten()
    
    res = {}
    
    if np.sum([return_rawCounts, return_normalizedCounts, return_model]) == 0:
        raise ValueError('No return specified. Set at least one out of return_RawCounts, return_normalizedCounts, return_model to True')
    
    if not naive:
    
        if np.any(counts_nuclei == 0) or np.any(np.isnan(counts_nuclei)) or np.any(np.isinf(counts_nuclei)):
            raise ValueError('Some of your nuclei counts are 0, nan or inf')

        if verbose:

            print('Initializing model...')
        
        if np.any(slide_id):
            model = ProbeCounts_GeneralModel_V2(
                X_data = counts_geneProbes,
                Y_data = counts_negativeProbes,
                nuclei = counts_nuclei,
                slide_id = slide_id,
                n_factors = n_factors)
            res['Model Name'] = 'Multi-Slide'
            if verbose:
                print('Using multi-slide model')
        else:
            model = ProbeCounts_GeneralModel(
                X_data = counts_geneProbes,
                Y_data = counts_negativeProbes,
                nuclei = counts_nuclei,
                n_factors = n_factors)
            res['Model Name'] = 'Single-Slide'
            if verbose:
                print('Using single-slide model')

        if verbose:

            print('Fitting model ...')

        model.fit_advi_iterative(n_iter = total_iterations, learning_rate = learning_rate, n=1, method='advi')

        if verbose:

            model.plot_history()
            plt.show()
            model.plot_history(iter_start = int(np.round(total_iterations - (total_iterations*0.1))),
                               iter_end = int(total_iterations))
            plt.show()

            print('Sampling from posterior distribution...')

        model.sample_posterior(node='all', n_samples=posterior_samples, save_samples=False);

        model.compute_X_corrected()          

    else:
        
        model = ProbeCounts_GeneralModel(
            X_data = counts_geneProbes,
            Y_data = counts_negativeProbes,
            nuclei = counts_nuclei,
            n_factors = n_factors)
        
        if verbose:
            print('Using naive model')
        
        res['Model Name'] = 'Naive'
        
        model.X_corrected_mean = np.round(np.clip(counts_geneProbes - np.mean(counts_negativeProbes, axis = 1).reshape(np.shape(counts_negativeProbes)[0],1), a_min = 0, a_max = None))

    if return_rawCounts:

        res['RawCounts'] = model.X_corrected_mean

    if return_normalizedCounts:

        total_counts = np.sum(model.X_corrected_mean, axis = 1)
        cpm = model.X_corrected_mean/total_counts.reshape(np.shape(model.X_corrected_mean)[0],1)*10**6
        res['NormCounts'] = cpm

    if return_model:

        res['Model'] = model
        
    if verbose:

        print('Done.')
                
    return res
              
