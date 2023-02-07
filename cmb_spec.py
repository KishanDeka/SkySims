import numpy as np
import camb


cosmos = camb.model.CAMBparams(WantTensors=True,max_l=8000,max_l_tensor=8000)


cosmos.set_cosmology(H0=67.63,omch2=0.12,ombh2=0.022,omk=0.0,tau=0.054)
cosmos.InitPower.set_params(As=2.1e-9,ns=0.965,r=0.001)

result = camb.get_results(cosmos)
result.calc_power_spectra(cosmos)

lmax = 7000
ps_arr = np.zeros((lmax+1,10))
ps_arr[:,0] = np.arange(lmax+1)

unl_cls = result.get_unlensed_total_cls(lmax=lmax,CMB_unit='muK')
ps_arr[:,1:5]=unl_cls

pot_cls = result.get_lens_potential_cls(lmax=lmax,CMB_unit='muK')
ps_arr[:,7:10]=pot_cls

np.savetxt('output/ps_out.txt',ps_arr)
del ps_arr,pot_cls,unl_cls

tot_cls = result.get_total_cls(lmax=7000,CMB_unit='muK')
np.savetxt('output/ps_len_theory.txt',tot_cls)
del tot_cls
