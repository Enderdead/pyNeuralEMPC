#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
 
import pyNeuralEMPC
 
setup(
    name='pyNeuralEMPC',
    version=pyNeuralEMPC.__version__,
    packages=find_packages(),
    author="François 'Enderdead' Gauthier-Clerc",
 
    author_email="francois@gauthier-clerc.fr",
 
    description="A nonlinear MPC library that allows you using a neural network as a model.",
 
    long_description=open('README.md').read(),
    
    install_requires=["numpy", "matplotlib", "tensorflow", "cyipopt", "jax"],

    include_package_data=True,
 
    url='https://github.com/Enderdead/pyNeuralEMPC',

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Neural NMPC",
    ],
    license="MIT",
)