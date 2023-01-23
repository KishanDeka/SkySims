import numpy as np
import camb


cosmo = camb.model.CAMBparams()

cosmo.set_cosmology(H0=67.63,omch2=0.12,ombh2=0.022,omk=0.0,tau=0.054)
cosmo.InitPower.set_params(As=2.1e-9,ns=0.965,r=0.0)

result = camb.get_results(cosmo)

result.save_cmb_power_spectra('ps_out.txt',lmax=2400)
