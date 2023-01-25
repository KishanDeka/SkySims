#!usr/env/python

import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt

indir = 'output/'
outdir = 'plots/'

#rel = sys.argv[1]

# read the maps
ulsfile = 'smooth_maps/ucmb_smooth.fits'
lnsfile = 'smooth_maps/lcmb_smooth.fits'
potfile = indir+'phi000_3.fits'
grdfile = indir+'grad000_3.fits'
spcfile = indir+'spec000_3.txt'

unl_map = hp.read_map(ulsfile,field=(0,1,2))
len_map = hp.read_map(lnsfile,field=(0,1,2))
len_pot = hp.read_map(potfile,field=0)
grd_pot = hp.read_map(grdfile,field=(0,1))


# calculate lensed and unlensed spectra
alm_unl  = hp.map2alm(smh_unl_map, use_weights=True, iter=1)
cl_unl = np.array(hp.alm2cl(alm_unl))
alm_len = hp.map2alm(smh_len_map, use_weights=True, iter=1)
cl_len = np.array(hp.alm2cl(alm_len))

# reshape and multiply l(l+1)/2*pi
if cl_unl.ndim == 1: cl_unl = np.reshape(cl_unl, [1,cl_unl.size])
n = cl_unl.shape[1]
l = np.arange(cl_unl.shape[1])
cl_unl[:,1:] *= l[1:n]*(l[1:n]+1)/(2*np.pi)

if cl_len.ndim == 1: cl_len = np.reshape(cl_len, [1,cl_len.size])
n = cl_len.shape[1]
l = np.arange(cl_len.shape[1])
cl_len[:,1:] *= l[1:n]*(l[1:n]+1)/(2*np.pi)

# plot full sky maps
fig1, (ax1,ax2) = plt.subplots(figsize=[12,6],ncols=2)

plt.axes(ax1)
hp.mollview(unl_map[0],unit='uK',title=r'unlensed T',max=500,min=-500,hold=True,cmap=plt.cm.jet)

plt.axes(ax2)
hp.mollview(len_map[0],unit='uK',title=r'lensed T',max=500, min=-500,hold=True,cmap=plt.cm.jet)

plt.savefig(outdir+'fullsky_temp.jpg',bbox_inches='tight')


fig2, ((ax1,ax2),(ax1,ax2)) = 





