#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:09:01 2018

@author: kalpanjasani
"""

import project
import getData

X, y = getData.getXY("../dataset/1000.csv")
project.makeFeatureNumberGraph(X, y, (1, 30), 2)
