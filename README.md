# InSituCellTools (isctools)
Methods for analysis of NanostringWTA and other probe based spatial transcriptomics data using probabilistic generative models and variational inference.

## Configure environment

You need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

Create conda environment with the required packages

```bash
conda create -n cellpymc python=3.7 numpy pandas jupyter python-igraph scanpy \
hyperopt cmake nose tornado dill ipython seaborn matplotlib request \
mkl-service pygpu --channel bioconda --channel conda-forge
```
Do not install pymc3 and theano with conda because it will not use the system cuda and we had problems with cuda installed in the local environment, install them with pip

```bash
pip install plotnine pymc3 torch pyro-ppl
```

## Notebooks

See [Part 1: Background Removal and Differential Expression with isctools](https://github.com/AlexanderAivazidis/InSituCellTools/blob/master/notebooks/Part1_BackgroundAndDifferentialExpression.ipynb) for removing background noise and performing differential expression.

See [Part 2: Cell Type Mapping with cell2location](https://github.com/AlexanderAivazidis/InSituCellTools/blob/master/notebooks/Part2_MapCelltypesToProbeCounts.ipynb) for cell type mapping.

The cell type mapping uses an adapted version of the cell2location algorithm (see [cell2location](https://github.com/BayraktarLab/cell2location)) and will soon be integrated with the cell2location package, but for now it is part of isctools.
