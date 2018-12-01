#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:09:01 2018

@author: kalpanjasani
"""
import matplotlib.pyplot as plt

import project
import getData

X, y = getData.getXY("../dataset/1000.csv")
project.makePCAGraph(X, y)
plt.show()
