from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import io
import os, sys, time, re, json

imread = plt.imread

color_to_label ={
    (0,0,0):0,
    (1,1,0):1,
    (1,0,0):2,
    (0,1,0):3,
    (0,0,1):4
}

def read_png(res):
    import PIL.Image
    img = PIL.Image.open(io.BytesIO(res))
    return np.asarray(img)

def read_npy(res):
    return np.load(io.BytesIO(res))

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, label in color_to_label.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3)*255, axis=2)
        arr_2d[m] = label
    return arr_2d