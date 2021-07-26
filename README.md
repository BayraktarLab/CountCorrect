# CountCorrect
A method for background removal from Nanostring WTA data

## Notebooks

See [Background Removal with CountCorrect](https://github.com/AlexanderAivazidis/CountCorrect/blob/main/BackgroundCorrection.ipynb)

## Installation

You will first need to configure an appropriate conda environment and then you can install countcorrect directly from github. How to do this is explained below.

### 1. Configure `countcorrect` conda environment

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```
Now there are two ways you can configure the 'countcorrect' conda environment:

#### 1. Method 1 (preferred): Create environment from file

Create `countcorrect` environment from file, which will install all the required conda and pip packages:

```bash
git clone https://github.com/BayraktarLab/CountCorrect/
cd CountCorrect
conda env create -f environment.yml
```

#### 1. Method 2: Create conda environment manually

Create conda environment manually using first conda:

```bash
conda create -n countcorrect python=3.7 numpy pandas jupyter scanpy \
ipython seaborn matplotlib \
pygpu --channel bioconda --channel conda-forge
```

And then install these packages with pip:

```bash
pip install "pymc3>=3.8,<3.10" torch arviz==0.11.1 adjustText
```

### 2. Install `countcorrect` package

Finally you can install countcorrect:

```bash
 pip install git+https://github.com/BayraktarLab/CountCorrect/
```
