from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import io
import os, sys, time, re, json

imread = plt.imread

def read_png(res):
    import PIL.Image
    img = PIL.Image.open(io.BytesIO(res))
    return np.asarray(img)

def read_npy(res):
    return np.load(io.BytesIO(res))