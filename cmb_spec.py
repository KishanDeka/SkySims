import numpy as np
import camb
from camb import correlations


cosmo = camb.model.CAMBparams()

cosmo.set_cosmology(H0=67.63,omch2=0.12,ombh2=0.022,omk=0.0,tau=0.054)
cosmo.InitPower.set_params(As=2.1e-9,ns=0.965,r=0.0)

result = camb.get_results(cosmo)
result.calc_power_spectra()

result.save_cmb_power_spectra('ps_out.txt',lmax=2400)

data = np.transpose(np.loadtxt('ps_out.txt'))
cls = np.transpose(data[1:5])
clpp = data[-1]
print(clpp)
#cl_lensed = correlations.lensed_cls(cls,clpp,lmax=2400,lmax_lensed=2400)
cl_lensed = result.get_lensed_cls_with-spectrum()
np.savetxt('ps_lens_theory.txt',cl_lensed)
