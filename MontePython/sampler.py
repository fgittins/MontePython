# -*- coding: utf-8 -*-
"""
A base sampler class.
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)
import numpy as np

class Sampler(object):
    """
    An abstract sampler object that implements basic helper functions
    required in most MCMC samplers.

    Parameters
    ----------

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
    """
    def __init__(self, dim, lnprobfn, args=[], kwargs={}):
        self.dim = dim
        self.lnprobfn = lnprobfn
        self.args = args
        self.kwargs = kwargs

        # Starting the random number generator
        self._random = np.random.mtrand.RandomState()

        self.reset()
    
    def reset(self):
        """
        Clear the ``chain``, ``lnprobability`` and reset the book-keeping
        parameters.
        """
        self._chain = np.empty((0, self.dim))
        self._lnprob = np.empty(0)
        
        self.iterations = 0
        self.accepted = 0
        self._last_run = None
    
    @property
    def random_state(self):
        """The state of the internal random number generator."""
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        """
        Tries to set the state of the random number generator. It fails 
        silently if it doesn't work.
        """
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def acceptance_fraction(self):
        """The fraction of the proposed steps that were accepted."""
        return self.accepted / self.iterations
    
    @property
    def chain(self):
        """Pointer to the Markov chain."""
        return self._chain
    
    @property
    def lnprobability(self):
        """
        List of log-probability values associated with each step in the chain.
        """
        return self._lnprob
    
    def get_lnprob(self, p):
        """The log-probability at the given position."""
        return self.lnprobfn(p, *self.args, **self.kwargs)
    
    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sampling routine must be implemented "
                                  "by subclasses")
    
    def run(self, p0, N, lnprob0=None, rstate0=None, **kwargs):
        """
        Run ``sample`` for ``N`` iterations and return the result.

        Parameters
        ----------

        p0 : list
            The initial position vector.
        N : int
            Number of iterations to run ``sample``.        
        lnprob0 : list (optional)
            The log-probability at position ``p0``.
        rstate0 : list (optional)
            The state of the random number generator.       
        kwargs : dict (optional)
            Other parameters that are provided directly to ``sample``.

        Returns
        -------

        results : tuple
            The final position vector, log-probability and state of the
            random number generator.
        
        Notes
        -----

        If ``p0`` is not provided, it will take the value from the last run.
        This will also be done for ``lnprob0`` and ``rstate0``.
        """
        if p0 is None:
            if self._last_run is None:
                raise ValueError("Cannot have p0=None if run has never "
                                 "been called.")
            p0 = self._last_run[0]
            if lnprob0 is None:
                lnprob0 = self._last_run[1]
            if rstate0 is None:
                rstate0 = self._last_run[2]
        
        for results in self.sample(p0, lnprob0, rstate0, iterations=N, 
                                   **kwargs):
            pass
        
        # Store for ``p0=None`` case
        self._last_run = results[:3]

        return results
