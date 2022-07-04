#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 07:55:18 2022

@author: mcuiller
"""

"""
A python package for Guassian processes regression.
This python packages contains a couple of useful classes to deal with basic
Gaussian process regression tasks.
The two classes implemented by this package are Kernel, with a number of
subclasses and GaussianProcess:
class GaussianProcess: the main class of the package.
module kernels: a module containing the Kernel class and its subclasses.
"""
from . import kernels
from .gaussianprocess import GaussianProcess

__all__ = ['GaussianProcess', 'kernels']

__author__ = 'Rodrigo F. Diaz'
