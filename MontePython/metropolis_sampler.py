# -*- coding: utf-8 -*-
"""
A Metropolis sampler.
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)
import numpy as np

from .sampler import Sampler

class MetropolisSampler(Sampler):
    """
    A basic implementation of the Metropolis algorithm.

    Parameters
    ----------

    cov : list or float
        The covariance matrix to use for the Gaussian proposal distribution.
    dim : int
        The number of dimensions in the parameter space.   
    lnprobfn : function
        A function that takes a vector in the parameter space as input and 
        returns the natural logarithm of the probability at that position.    
    args : list (optional)
        Positional arguments for ``lnprobfn``. ``lnprobfn`` will be called 
        as ``lnprobfn(p, *args, **kwargs)``.    
    kwargs : dict (optional)
        Keyword arguments for ``lnprobfn``. ``lnprobfn`` will be called as 
        ``lnprobfn(p, *args, **kwargs)``.
    
    Notes
    -----

    ``cov`` can be a list or a float depending on whether the problem is
    multidimensional or one-dimensional. In the one-dimensional case, 
    ``cov`` is interpreted as the variance of the Gaussian proposal 
    distribution.
    """
    def __init__(self, cov, *args, **kwargs):
        super(MetropolisSampler, self).__init__(*args, **kwargs)
        self.cov = cov
    
    def sample(self, p, lnprob=None, rstate=None, thin=1, 
               store_chain=True, iterations=1):
        """
        Advances the the chain ``iterations`` steps as an iterator.

        Parameters
        ----------

        p : list
            The starting position vector.
        lnprob : list (optional)
            The log-probability at the starting position. If not provided, the
            values are calculated.
        rstate : list (optional)
            The state of the random number generator.
        thin : int (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set ``thin`` to an integer greater than 1.
        store_chain : bool (optional)
            By default, the sampler stores the positions and log-probabilities 
            of the samples in the chain.      
        iterations : int (optional)
            The number of steps to run.
        
        Returns
        -------

        p : list
            The accepted position vector.        
        lnprob : list
            The log-probability at the accepted position vector.        
        rstate : list
            The state of the random number generator.
        
        Notes
        -----

        If ``rstate`` is not provided the setting of the ``random_state``
        will fail silently and use the initial ``random_state``.
        """
        # This will fail silently if ``rstate=None`` and the initial
        # ``random_state`` will be used
        self.random_state = rstate

        p = np.asarray(p)
        if lnprob is None:
            lnprob = self.get_lnprob(p)
        
        if store_chain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                                          np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))
        
        i0 = self.iterations
        for i in range(iterations):
            self.iterations += 1

            if self.dim == 1:
                q = self._random.normal(p, np.sqrt(self.cov))
            else:
                q = self._random.multivariate_normal(p, self.cov)
            newlnprob = self.get_lnprob(q)
            diff = newlnprob - lnprob

            if diff < 0:
                diff = np.exp(diff) - self._random.rand()
            
            if diff > 0:
                p = q
                lnprob = newlnprob
                self.accepted += 1
            
            if store_chain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob
            
            yield p, lnprob, self.random_state
