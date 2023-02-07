#!usr/env/python

import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt

indir = 'outputs/'
outdir = 'plots/'

#rel = sys.argv[1]

# read the maps
ulsfile = indir+'ucmb_smooth.fits'
lnsfile = indir+'lcmb_smooth.fits'
potfile = indir+'phi000_3.fits'
grdfile = indir+'grad000_3.fits'
spcfile = indir+'spec000_3.txt'

#unl_map = hp.read_map(ulsfile,field=(0,1,2))
#len_map = hp.read_map(lnsfile,field=(0,1,2))
#len_pot = hp.read_map(potfile,field=0)
#grd_pot = hp.read_map(grdfile,field=(0,1))


##################################################################################
####################   SKY MAPS      ##########################################
##################################################################################


# plot full sky maps
#fig1, (ax1,ax2) = plt.subplots(figsize=[12,6],ncols=2)

#plt.axes(ax1)
#hp.mollview(unl_map[0],unit='uK',title=r'unlensed T',max=500,min=-500,hold=True,cmap=plt.cm.jet)

#plt.axes(ax2)
#hp.mollview(len_map[0],unit='uK',title=r'lensed T',max=500, min=-500,hold=True,cmap=plt.cm.jet)

#plt.savefig(outdir+'fullsky_temp.pdf',bbox_inches='tight')


## plot lensing and unlensing difference
#titles= ['T','Q','U']
#xs = 1000
#cen = [20,20]
#mval = [50,4,4]

#for i in range(3):
#	fig2, ((ax1, ax2),(ax3,ax4)) = plt.subplots(figsize=[10,12],ncols=2,nrows=2)
#	
#	plt.axes(ax1)
#	hp.gnomview(unl_map[i],unit='uK',rot=cen,xsize=xs,ysize=xs,title='unlensed %s' %titles[i],hold=True,cmap=plt.cm.jet)

#	plt.axes(ax2)
#	hp.gnomview(len_map[i],unit='uK',rot=cen,xsize=xs,ysize=xs,title='lensed %s' %titles[i],hold=True,cmap=plt.cm.jet)

#	plt.axes(ax3)
#	hp.gnomview(unl_map[i]-len_map[i],rot=cen,max=mval[i],min=-mval[i],unit='uK',xsize=xs,ysize=xs,title='difference',hold=True,cmap=plt.cm.jet)

#	plt.axes(ax4)
#	hp.gnomview(np.linalg.norm(grd_pot,axis=0),rot=cen,unit='uK',xsize=xs,ysize=xs,title='grad of lens potential',hold=True,cmap=plt.cm.jet)
#	
#	plt.savefig(outdir+'lensing_pot_recon_%s.pdf' %titles[i],bbox_inches='tight')


#unl_pol = np.sqrt(unl_map[1]**2+unl_map[2]**2)
#len_pol = np.sqrt(len_map[1]**2+len_map[2]**2)

#fig2, ((ax1, ax2),(ax3,ax4)) = plt.subplots(figsize=[10,12],ncols=2,nrows=2)

#plt.axes(ax1)
#hp.gnomview(unl_pol,unit='uK',rot=cen,xsize=xs,ysize=xs,title='unlensed pol deg',hold=True,cmap=plt.cm.jet)

#plt.axes(ax2)
#hp.gnomview(len_pol,unit='uK',rot=cen,xsize=xs,ysize=xs,title='lensed pol deg',hold=True,cmap=plt.cm.jet)

#plt.axes(ax3)
#hp.gnomview(unl_pol-len_pol,rot=cen,max=3,min=-3,unit='uK',xsize=xs,ysize=xs,title='diff',hold=True,cmap=plt.cm.jet)

#plt.axes(ax4)
#hp.gnomview(np.linalg.norm(grd_pot,axis=0),rot=cen,unit='uK',xsize=xs,ysize=xs,title='grad of lens potential',hold=True,cmap=plt.cm.jet)

#plt.savefig(outdir+'lensing_pot_recon_pol.pdf',bbox_inches='tight')


##################################################################################
####################   POWER SPECTRA    ##########################################
##################################################################################


# calculate lensed and unlensed spectra 
#alm_unl  = hp.map2alm(unl_map, use_weights=True, iter=1)
#cl_unl = np.array(hp.alm2cl(alm_unl))
#alm_len = hp.map2alm(len_map, use_weights=True, iter=1)
#cl_len = np.array(hp.alm2cl(alm_len))

## reshape and multiply l(l+1)/2*pi
#if cl_unl.ndim == 1: cl_unl = np.reshape(cl_unl, [1,cl_unl.size])
#n = cl_unl.shape[1]
#l = np.arange(cl_unl.shape[1])
#cl_unl[:,1:] *= l[1:n]*(l[1:n]+1)/(2*np.pi)

#if cl_len.ndim == 1: cl_len = np.reshape(cl_len, [1,cl_len.size])
#n = cl_len.shape[1]
#l = np.arange(cl_len.shape[1])
#cl_len[:,1:] *= l[1:n]*(l[1:n]+1)/(2*np.pi)

#np.save(indir+'unl_ps_noisy.npy',cl_unl)
#np.save(indir+'len_ps_noisy.npy',cl_len)


cl_unl = np.load(indir+'unl_ps.npy')
cl_len = np.load(indir+'len_ps.npy')

# get theoretical spectra

ps_unl_theo = np.transpose(np.loadtxt('ps_out.txt'))
ell1 = ps_unl_theo[0]
bm_gauss = hp.gauss_beam(fwhm=5*np.pi/60/180,lmax=len(ell1)-1,pol=False)
TT_unl = ps_unl_theo[1]*(bm_gauss)**2
EE_unl = ps_unl_theo[2]*(bm_gauss)**2
BB_unl = ps_unl_theo[3]*(bm_gauss)**2

ps_len_theo = np.transpose(np.loadtxt('ps_lens_theory.txt'))
#ell = ps_len_theo[0]
bm_gauss = hp.gauss_beam(fwhm=5*np.pi/60/180,lmax=len(ell1)-1,pol=False)
TT_len = ps_len_theo[0]*(bm_gauss)**2
EE_len = ps_len_theo[1]*(bm_gauss)**2
BB_len = ps_len_theo[2]*(bm_gauss)**2


fig, (ax1,ax2,ax3) = plt.subplots(figsize=[15,4],ncols=3,nrows=1)

ell = np.arange(cl_len.shape[1])
#ax1.plot(ell,cl_unl[0],label='unl TT')
#ax1.plot(ell,cl_len[0],label='len TT')
#ax1.fill_between(ell1,TT_unl*(1-np.sqrt(2/(2*ell1+1))),TT_unl*(1+np.sqrt(2/(2*ell1+1))))
#ax1.fill_between(ell1,TT_unl*(1-2*np.sqrt(2/(2*ell1+1))),TT_unl*(1+2*np.sqrt(2/(2*ell1+1))))
ax1.plot(ell1,TT_unl,'k--',label='theo unl TT')
ax1.plot(ell1,TT_len,'r-.',label='theo len BB')
ax1.set_title('TT')
ax1.legend()
ax1.set_xlim(0,1500)

ax2.plot(ell,cl_unl[1],label='unl EE')
ax2.plot(ell,cl_len[1],label='len EE')
ax2.plot(ell1,EE_unl,'k--',label='theo unl EE')
ax2.set_title('EE')
ax2.legend()
ax2.set_xlim(600,1700)

ax3.plot(ell,cl_unl[2],label='unl BB')
ax3.plot(ell,cl_len[2],label='len BB')
ax3.plot(ell1,BB_unl,'k--',label='theo unl BB')
ax3.plot(ell1,BB_len,'r--',label='theo len BB')
ax3.set_xlim(0,2000)
ax3.legend()
ax3.set_title('BB')

plt.savefig(outdir+'power_spec.pdf',bbox_inches='tight')


# plot relative difference



