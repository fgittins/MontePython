# MontePython
A Markov chain-Monte Carlo (MCMC) sampler written in Python. The current implementation is based on [emcee](https://github.com/dfm/emcee).

## getting started.
In order to run the code, all that is needed is `numpy`. This can be installed
using `conda`:
```
conda install numpy
```

## basic usage.
Here is a simple example of sampling from a 10-dimensional Gaussian using the 
Metropolis method:
```
import numpy as np
import montepython

def lnprob(x, ivar):
    return -0.5 * np.sum(x**2 * ivar)

ndim = 10
ivar = 1 / np.random.rand(ndim)
cov = np.random.rand(ndim, ndim)
p0 = np.random.rand(ndim)

sampler = montepython.MetropolisSampler(cov, ndim, lnprob, args=[ivar])
sampler.run(p0, 1000)
```
