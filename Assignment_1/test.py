#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:14:40 2017

@author: Flore
"""

data = loadtxt('data.csv', delimiter=',')
y = array(data[:,0:1])
X = data[: ,1:data.shape[1]]

cst = X.min()

X0 = X + abs(cst)

print(X0.min())

