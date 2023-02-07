#!usr/env/python

import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt

unl_map = hp.read_map('outputs/ucmb_smooth.fits',field=(0,1,2))
len_map = hp.read_map('outputs/lcmb_smooth.fits',field=(0,1,2))
noise_map = hp.read_map('noise_maps/ffp10_noise_143_full_map_mc_00000.fits',field=(0,1,2))

unl_map += noise_map
len_map += noise_map

hp.write_map("outputs/ucmb_smooth_noisy.fits", unl_map,overwrite=True)
hp.write_map("outputs/lcmb_smooth_noisy.fits", len_map,overwrite=True)
