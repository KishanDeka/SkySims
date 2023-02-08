#!usr/env/python

import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt

indir = 'outputs/'
outdir = 'plots/'

#rel = sys.argv[1]

# read the maps
#ulsfile = indir+'ucmb_smooth.fits'
#lnsfile = indir+'lcmb_smooth.fits'
#potfile = indir+'phi000_3.fits'
#grdfile = indir+'grad000_3.fits'
#spcfile = indir+'spec000_3.txt'

#unl_map = hp.read_map(ulsfile,field=(0,1,2))
#len_map = hp.read_map(lnsfile,field=(0,1,2))
#len_pot = hp.read_map(potfile,field=0)
#grd_pot = hp.read_map(grdfile,field=(0,1))


##################################################################################
####################   SKY MAPS      ##########################################
##################################################################################


##  plot full sky maps

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


##################################################################################
####################   POWER SPECTRA    ##########################################
##################################################################################

cl_uls = np.loadtxt(indir+'ucmb_cls_smooth.txt')
cl_lns = np.loadtxt(indir+'lcmb_cls_smooth.txt')

# get theoretical spectra

ps_uls_theo = np.transpose(np.loadtxt(indir+'ps_out.txt'))
ell1 = ps_uls_theo[0]
bm_gauss1 = hp.gauss_beam(fwhm=5*np.pi/60/180,lmax=len(ell1)-1,pol=False)


ps_lns_theo = np.transpose(np.loadtxt(indir+'ps_lns_theory.txt'))
ell2 = np.arange(ps_lns_theo.shape[1])
bm_gauss2 = hp.gauss_beam(fwhm=5*np.pi/60/180,lmax=len(ell2)-1,pol=False)



fig, axes = plt.subplots(figsize=[10,10],ncols=2,nrows=2)

titles = ['TT','EE','BB','TE']
ell = np.arange(cl_lns.shape[1])

for i in range(4):
	ax = axes.flatten()[i]
	ax.plot(ell,cl_uls[i],label='unl %s' %titles[i])
#	ax.plot(ell,cl_lns[i],label='len %s' %titles[i])
	
	# plot unlensed theory with variance
	ucl = ps_uls_theo[i+1]*(bm_gauss1)**2
	ax.plot(ell1,ucl,'k--')
	ax.fill_between(ell1,ucl*(1-np.sqrt(2/(2*ell1+1))),ucl*(1+np.sqrt(2/(2*ell1+1))))
	ax.fill_between(ell1,ucl*(1-2*np.sqrt(2/(2*ell1+1))),ucl*(1+2*np.sqrt(2/(2*ell1+1))))
	
	
	# plot lensed theory with variance
#	lcl = ps_lns_theo[i]*(bm_gauss2)**2
#	ax.plot(ell1,ucl,'k--')
#	ax.fill_between(ell2,lcl*(1-np.sqrt(2/(2*ell2+1))),lcl*(1+np.sqrt(2/(2*ell2+1))))
#	ax.fill_between(ell2,lcl*(1-2*np.sqrt(2/(2*ell2+1))),lcl*(1+2*np.sqrt(2/(2*ell2+1))))
	
	ax.set_title(titles[i])
	ax.legend()
	ax.set_xlim(200,1300)

plt.savefig(outdir+'power_spec.pdf',bbox_inches='tight')


# plot relative difference



