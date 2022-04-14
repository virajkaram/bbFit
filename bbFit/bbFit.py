from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve
from astropy.convolution import Box1DKernel
import matplotlib
from astropy.table import Column
from astropy.time import Time
from astropy.coordinates import Distance
from astropy.cosmology import Planck18, Planck18
import emcee
import corner
from emcee.autocorr import AutocorrError


class BBFit:

    def mag2fnu(self):
        Jy = 1e-23 #erg/s/cm2/Hz
        self.fnu = 10**(-0.4*self.mags)*self.zpt *Jy
        self.fnuerr = self.magerrs*self.fnu * np.log(10)/2.5


    def calc_Lnu(self):
        self.Lnu = self.fnu* 4*np.pi*self.dist**2
        self.Lnuerr = self.fnuerr*4*np.pi*self.dist**2 

        if self.disterr>0:
            self.Lnuerr = np.sqrt(self.Lnuerr**2 + (4*np.pi*self.fnu*2*self.dist*self.disterr)**2)

        return 1


    def ext_corr(self):
        if np.any(self.As < 0):
            print('Please enter a valid value for extinction A')
            #raise ValueError()
            return -99

        ecor = 10**(-0.4*self.As)
        self.fnu = self.fnu/ecor
        self.fnuerr = self.fnuerr/ecor
        self.Lnu = self.Lnu/ecor
        self.Lnuerr = self.Lnuerr/ecor


    def blackbody_lambda(self,T,wavs):
    #wavs in cm
        h = 6.6e-27
        c = 3e10
        kb = 1.38e-16
        return (2*h*c**2/(wavs**5))*(1/(np.exp(h*c/(wavs*kb*T)) - 1))

    
    def bb_fit(self,Teff,Rphot,dist,wavs):
        #wavs in cm
        #Rphot, dist in cm
        #return lambda*f_lambda
        Lsun = 3.83e33 #erg/s
        return np.pi*((Rphot/dist)**2) * wavs* self.blackbody_lambda(Teff,wavs)

    
    def uniform_prior(self,p,pars):
        pmin,pmax = pars
        if pmin<p<pmax:
            return 1/(pmax-pmin)
        
        else:
            return 0
        
    
    def prior(self,p,pars,pri_type = 'uniform'):
        if pri_type=='uniform':
            return self.uniform_prior(p,pars)
    
    
    def log_likelihood(self,theta, wavs, l, lerr, dist, pri_type, Tpripars, Rpripars,prilog=False):
        #dist : defined outside the function
        if prilog:
            Teff,Rphot = 10**theta
        else:
            Teff,Rphot = theta
        model = self.bb_fit(Teff,Rphot,dist,wavs)
        sigma2 = lerr ** 2
        pri = self.prior(theta[0],Tpripars,pri_type)*self.prior(theta[1],Rpripars,pri_type)
        
        lims = (self.magerrs < 0)
        if len(self.mags[lims])>0:
            if np.any((model[lims] - l[lims])>0):
                return -np.inf

        l = l[np.invert(lims)]
        model = model[np.invert(lims)]
        sigma2 = sigma2[np.invert(lims)]

        if pri==0:
            return -np.inf
        return -0.5 * np.sum((l - model) ** 2 / sigma2 + np.log(sigma2)) + np.log10(pri)
    

    def mcmc_fit(self,pri_type=None,Tpripars=None,Rpripars=None,pos=None,nchain=50000,plot=True,prilog=False):
        if pos is None:
            pos = (6000,1e13) + (100,2e11)*np.random.randn(32, 2)

        
        nwalkers, ndim = pos.shape

        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_likelihood, args=(self.wavs, self.nus*self.fnu, self.nus*self.fnuerr, self.dist, pri_type, Tpripars, Rpripars,prilog))
        sampler.run_mcmc(pos, nchain, progress=True)
        samples = sampler.get_chain()

        try:
            tau = sampler.get_autocorr_time()
            print('Autocorrelation length',tau)

        except AutocorrError as e:
            print('Warning',e,'Setting tau to %i'%(nchain/50))
            tau = [nchain/50,nchain/50]
        #return sampler
        flat_samples = sampler.get_chain(discard=int(10*np.max(tau)), thin=15,flat=True)

        Ts = flat_samples[:,0]
        Rs = flat_samples[:,1]
        sb = 5.67e-5 #Stefan Boltzmann
        if prilog:
            Ls = np.log10(4*np.pi*sb) + 2*Rs + 4*Ts
        else:
            Ls = (4*np.pi*Rs**2)*sb*Ts**4

        flat_samples = np.reshape(np.array([Ts,Rs,Ls]).T,(len(Ts),3))

        if plot:
            self.corner_plot(flat_samples)

        self.flat_samples = flat_samples
        return flat_samples


    def corner_plot(self,flat_samples):
        fig = corner.corner(flat_samples, labels=self.labels)
        plt.savefig('%s/%s_corner.pdf'%(self.plotdir,self.name),bbox_inches='tight')


    def fits_plot(self,flat_samples):
        inds = np.random.randint(len(flat_samples), size=500)
        plot_wavs = np.linspace(0.4,2,100)*1e-4
        plt.figure()
        for ind in inds:
            sample = flat_samples[ind]
            plt.plot(plot_wavs*1e4, bb_fit(sample[0],sample[1],self.dist,plot_wavs),'C1',alpha=0.1)
        
        plt.plot(wavs*1e4,nus*fnu,'.',c='black')
        plt.errorbar(wavs*1e4,nus*fnu,nus*fnuerr,fmt='.',c='black')
        plt.yscale('log')
        plt.xscale('log')
        #plt.xlim(0.5)
        plt.xlabel(r'$\lambda$ [$\mu$m]')
        plt.ylabel(r'$\lambda$f$_\lambda$ [erg/s/cm$^{2}$/Hz]')
        plt.savefig('%s/%s_fits.pdf'%(self.plotdir,self.name),bbox_inches='tight')


    def plot_init(self):
        matplotlib.rcParams['xtick.minor.size'] = 6
        matplotlib.rcParams['xtick.major.size'] = 6
        matplotlib.rcParams['ytick.major.size'] = 6
        matplotlib.rcParams['ytick.minor.size'] = 6
        matplotlib.rcParams['lines.linewidth'] = 1.5
        matplotlib.rcParams['axes.linewidth'] = 1.5
        matplotlib.rcParams['font.size']= 16
        matplotlib.rcParams['font.family']= 'sans-serif'
        matplotlib.rcParams['xtick.major.width']= 2.
        matplotlib.rcParams['ytick.major.width']= 2.
        matplotlib.rcParams['ytick.direction']='in'
        matplotlib.rcParams['xtick.direction']='in'


    def print_estimates(self,scales=None):
        results = []
        ndim = self.flat_samples.shape[1]

        Rsun = 7e10 #cm
        Lsun = 3.83e33 #erg/s
        if scales is None:
            scales = [1,Rsun,Lsun]
        for i in range(ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [5, 50, 95])
            results.append(mcmc)
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1]/scales[i], q[0]/scales[i], q[1]/scales[i], self.labels[i])
            print(txt)


    def __init__(self,mags,magerrs,wavs,zpt,dist=-99,disterr=-99,As=np.array([-99,-99]),name='seds',plotdir='.'):
        self.mags = mags
        self.magerrs = magerrs
        self.wavs = wavs
        self.zpt = zpt
        self.dist = dist #Mpc
        self.disterr = disterr #Mpc
        self.As = As
        c = 3e10 #cm/s
        self.nus = c/self.wavs

        self.fnu, self.fnuerr = -99, -99
        self.Lnu, self.Lnuerr = -99, -99

        self.name = name
        self.flat_samples = None
        self.labels = ["$T_{eff}$", "R", "L"]
        self.plotdir = plotdir

        self.plot_init()
        
