# CountCorrect
A method for background removal from Nanostring WTA data

## Notebooks

See [Background Removal with CountCorrect](https://github.com/AlexanderAivazidis/CountCorrect/blob/main/BackgroundCorrection.ipynb)

## Setting up a conda environment and installation:

Running CountCorrect on GPU is much faster and only takes 2-3 minutes for 100 samples compared to 2-3 hours on CPU.
We recommend to first set up a conda environment with the required packages, but not with pymc3 and theano:

```bash
conda create -n countcorrect python=3.7 numpy pandas jupyter scanpy \
ipython seaborn matplotlib \
pygpu --channel bioconda --channel conda-forge
```

And then install these packages with pip:

```bash
pip install "pymc3>=3.8,<3.10" torch arviz==0.11.1 adjustText
```
Finally you can install CountCorrect:

```bash
 pip install git+https://github.com/AlexanderAivazidis/CountCorrect/
```
