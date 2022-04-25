# bbFit
Python module for MCMC blackbody fits to multi-band photometric measurements of astronomical objects

Usage : 

mags = np.array([19,20])
magerrs = np.array([0.1,0.1])
wav_names = np.array(['g','r'])
wavs = np.array([0.46,0.66])*1e-4 #wavelength in cm
As = np.array([Ag, Ar]) # Ag, Ar from NED
zpt = np.array([3631]*len(wavs)) #zeropoint in Jy of the photometric system used (3631 for AB)
dist = xx # cm, Optional, but required if calculating luminosity and not just temperature
disterr = xx # cm Optional

bb = BBFit(mags,magerrs,wavs,zpt,dist,disterr=disterr,As=As)
bb.mag2fnu()
bb.ext_corr()
bb.calc_Lnu()

#Fit 2-dim BB model to data
pri_type = 'uniform'
Tpripars = (3.0,4.5) #Prior in log space
Rpripars = (10,16) #Prior in log space
pos = (3.5,13) + (0.05,0.1)*np.random.randn(32, 2)
flat_samples = bb.mcmc_fit(pri_type,Tpripars,Rpripars,nchain=10000,pos=pos,prilog=True)
    
bb.print_estimates()
bbfits_temp_estimates = list(np.percentile(10**flat_samples[:,0],[5,50,95]))
bbfits_rs_estimates = list(np.percentile(10**flat_samples[:,1],[5,50,95]))
bbfits_lums = list(np.percentile(10**flat_samples[:,2],[5,50,95]))
