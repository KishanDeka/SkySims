import numpy as np
import healpy as hp

dir = 'outputs/'

# smoothing the maps
def smoothing_maps(infile,outfile,fwhm,pol=True):
	if pol :
		m = hp.read_map(infile,field=(0,1,2))
	else :	m = hp.read_map(infile,field=(0))
	
	smooth_map = hp.smoothing(m,fwhm=fwhm,pol=pol,use_weights=True)
	hp.write_map(outfile,smooth_map,overwrite=True)
	del smooth_map,m
	
def calculate_cls(infile,outfile):
	m = hp.read_map(infile,field=(0,1,2))
	alm  = hp.map2alm(m, use_weights=True, iter=1)
	cl = np.array(hp.alm2cl(alm))

	## reshape and multiply l(l+1)/2*pi
	if cl.ndim == 1: cl = np.reshape(cl, [1,cl.size])
	n = cl.shape[1]
	l = np.arange(cl.shape[1])
	cl[:,1:] *= l[1:n]*(l[1:n]+1)/(2*np.pi)
	
	np.savetxt(outfile,cl)
	del cl,alm,m
	
def add_noise(infile,outfile,noisefile):
	m = hp.read_map(infile,field=(0,1,2))
	noise = hp.read_map(noisefile,field=(0,1,2))
	m += noise
	hp.write_map(outfile,m)
	
	
arr = ['ucmb','lcmb']
noisef = dir+'ffp10_noise_143_full_map_mc_00000.fits'
for s in arr:
	dis = dir+s
	smoothing_maps(dis+'000_3.fits',dis+'_smooth.fits',5*np.pi/60/180)
	calculate_cls(dis+'000_3.fits',dis+'_cls.txt')
	calculate_cls(dis+'_smooth.fits',dis+'_cls_smooth.txt')
	add_noise(dis+'_smooth.fits',dis+'_smooth_noisy.fits',noisef)
