# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name="montepython",
    version="0.1dev",
    author="Fabian Gittins",
    packages=["montepython"],
    license="MIT",
    description=("A Markov chain-Monte Carlo (MCMC) sampler written in "
                 "Python."),
    long_description=open("README.md").read(),
    install_requires=["numpy"],
)