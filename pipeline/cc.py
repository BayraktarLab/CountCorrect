import sys,os
import click
import pickle
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import diffxpy.api as de
from IPython.display import Image
data_type = 'float32'
os.environ["THEANO_FLAGS"] = 'device=cpu,floatX=' + data_type + ',force_device=True' + ',dnn.enabled=False'
import countcorrect as cc
from countcorrect.ProbeCounts__GeneralModel import ProbeCounts_GeneralModel


@click.command()
@click.option(
 '--genes',
 help = 'Full path to CSV file containing count matrix for gene probes',
 type = click.Path(exists = True),
 required = True
)
@click.option(
 '--negs',
 help = 'Full path to CSV file containing count matrix for negative control probes',
 type = click.Path(exists = True),
 required = True
)
@click.option(
 '--nuclei',
 help = 'Full path to file containing the number of nuclei per ROI, one per line, in the same order as the other files',
 type = click.Path(exists = True),
 required = True
)
@click.option(
 '--n_iter',
 help = 'Number of iterations for fitting the model',
 type = click.INT,
 required = True
)
def remove_background(genes, negs, nuclei, n_iter):
  click.echo('Gene probes path: ' + genes)
  click.echo('Negative probe path: ' + negs)
  click.echo('Nuclei counts: ' + nuclei)
  click.echo('Number of iterations: ' + str(n_iter))
  click.echo("Loading data...")
  geneProbes = np.genfromtxt(
    genes,
    delimiter = ',',
    dtype = 'float32'
    )
  click.echo("Loaded gene probes: " + str(geneProbes.shape))
  negProbes = np.genfromtxt(
    negs,
    delimiter = ',',
    dtype = 'float32'
    )
  click.echo("Loaded negative probes: " + str(negProbes.shape))
  nuclei = np.genfromtxt(
    nuclei,
    delimiter = ',',
    dtype = 'float32'
    )
  click.echo("Loaded nuclei counts: " + str(nuclei.shape))
  plt.ioff()
  click.echo("Creating model ...")
  model = ProbeCounts_GeneralModel(
    X_data = geneProbes,
    Y_data = negProbes,
    nuclei = nuclei,
    data_type = 'float32',
    n_factors = 30
    )
  click.echo("Fitting model ...")
  model.fit_advi_iterative(
    n_iter = n_iter,
    learning_rate = 0.01,
    n=1,
    method='advi'
    )
  plt.savefig("Figure0")
  click.echo("Sampling posterior ...")
  model.sample_posterior(
    node='all',
    n_samples=100,
    save_samples=True
    )
  click.echo("Computing correctons ...")
  model.compute_X_corrected()
  click.echo("Saving to CSV ...")
  np.savetxt(
    "corrected_counts.csv",
    model.X_corrected_mean,
    delimiter=","
    )
  click.echo("Plotting ...")
  model.plot_X_corrected_overview2()
  plt.savefig("Figure1")
  model.plot_X_corrected_overview3(saveFig = "Figure2.png")
  model.plot_X_corrected_overview5(['g_1','g_2'], saveFig = "Figure3.png")
  click.echo("Ranking ...")
  model.rank_X_corrected_genes().to_csv('ranked_correctons.csv')
