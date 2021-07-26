# CountCorrect
A method for background removal from Nanostring WTA data

## Notebooks

See [Background Removal with CountCorrect](https://github.com/AlexanderAivazidis/CountCorrect/blob/main/BackgroundCorrection.ipynb)

## Configure your own conda environment and installation

1. Installation of dependecies and configuring environment (Method 1 (preferred) and Method 2)
2. Installation of cell2location

Prior to installing countcorrect you need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

#### 1. Method 1 (preferred): Create environment from file

Create `countcorrect` environment from file, which will install all the required conda and pip packages:

```bash
git clone https://github.com/BayraktarLab/CountCorrect/
cd CountCorrect
conda env create -f environment.yml
```

#### 1. Method 2: Create conda environment manually

Create conda environment with the required packages pymc3 and scanpy:

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
