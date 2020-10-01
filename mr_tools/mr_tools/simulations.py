import galsim
from astropy import wcs as WCS
import numpy as np
import scipy.signal as scp
import scarlet
from scarlet_extensions.initialization.detection import Data
import pickle
import matplotlib.pyplot as plt
from . import galsim_compare_tools as gct

def load_surveys():
    """Creates dictionaries for the HST, EUCLID, WFRIST, HCS anf LSST surveys
    that contain their names, pixel sizes and psf fwhm in arcseconds"""
    pix_ROMAN = 0.11
    pix_RUBIN = 0.2
    pix_HST = 0.06
    pix_EUCLID = 0.101
    pix_HSC = 0.167

    #Sigma of the psf profile in arcseconds.
    sigma_ROMAN = [1.69*0.11] #https://arxiv.org/pdf/1702.01747.pdf Z-band
    sigma_RUBIN = [0.327, 0.31, 0.297, 0.285, 0.276, 0.267] #https://www.lsst.org/about/camera/features
    sigma_EUCLID = [0.16] #https://sci.esa.int/documents/33859/36320/1567253682555-Euclid_presentation_Paris_1Dec2009.pdf
    sigma_HST = [0.074] #Source https://hst-docs.stsci.edu/display/WFC3IHB/6.6+UVIS+Optical+Performance#id-6.6UVISOpticalPerformance-6.6.1 800nm
    sigma_HSC = [0.306, 0.285, 0.238, 0.268, 0.272] #https://hsc-release.mtk.nao.ac.jp/doc/ deep+udeep


    EUCLID = {'name': 'EUCLID',
              'pixel': pix_EUCLID ,
              'psf': sigma_EUCLID,
              'channels': ['VIS'],
              'sky':np.array([22.9]),
              'exp_time': np.array([2260]),
              'zero_point': np.array([6.85])}
    HST = {'name': 'HST',
           'pixel': pix_HST,
           'psf': sigma_HST,
           'channels': ['f814w'],
           'sky':None,
           'exp_time': None,
           'zero_point': None}
    HSC = {'name': 'HSC',
           'pixel': pix_HSC,
           'psf': sigma_HSC,
           'channels': ['g','r','i','z','y'],
           'sky': np.array([21.4, 20.6, 19.7, 18.3, 17.9]),
           'exp_time': np.array([600, 600, 1200, 1200, 1200]),
           'zero_point': np.array([91.11, 87.74, 69.80, 29.56, 21.53])}
    ROMAN = {'name': 'ROMAN',
             'pixel': pix_ROMAN,
             'psf': sigma_ROMAN,
             'channels': ['R062', 'Z087', 'Y106', 'J129', 'H158', 'F184'],
             'sky':None,
             'exp_time': None,
             'zero_point': None}
    RUBIN = {'name': 'RUBIN',
             'pixel': pix_RUBIN,
             'psf': sigma_RUBIN,
             'channels': ['u','g','r','i','z','y'],
             'sky': np.array([22.9, 22.3, 21.2, 20.5, 19.6, 18.6]),
             'exp_time': np.array([1680, 2400, 5520, 5520, 4800, 4800]),
             'zero_point': np.array([9.16, 50.70, 43.70, 32.36, 22.68, 10.58])}

    return HST, EUCLID, ROMAN, HSC, RUBIN

HST, EUCLID, ROMAN, HSC, RUBIN = load_surveys()

class Simulation:
    """ This class generates simulatiionis of patches with realistic galsim profiles.
    It then run scarlet on a set of scarlet_extensoinis.runner objects that take the

    parameters
    ----------
    cat: catalog
        Catalog to querey in order to extract galsim galaxies
    ngal: int
        The maximum number of sources allowed per patch
    runners: list of 'scarlet_extensions.runner's
        list of iinitialised runners to run scarlet
    cats: list of booleans
        list of boolean with the same length as runners. tells wehther to use the true catalog on the runner
        or to run the detection algorithm.
    hr_dict: dict
        What survey should be used to model the high resolution channel
    lr_dict: dict
        What survey should be used to model the low resolution channel
    n_lr: int
        Number of pixels on a side for the low resolution channel

    """
    def __init__(self, cat, runners, ngal = 10, cats = None, hr_dict=EUCLID, lr_dict=RUBIN, n_lr=60):

        self.runners = runners
        self.ngal = ngal
        self.n_lr = n_lr
        self.hr_dict = hr_dict
        self.lr_dict = lr_dict
        self.n_hr = np.int(np.around(self.n_lr*self.lr_dict['pixel']/self.hr_dict['pixel'], decimals = 3))
        if cats is not None:
            assert len(cats) == len(runners), 'cats should have the same length as runners'
            self.cats = cats
        else:
            self.cats = [False, False, False]
        results = []
        for r in runners:
            results.append( {'resolution': [],
                        'chi': [] ,
                        'SDRs': [],
                        'SED_SDRs': [],
                        'n_sources': [],
                        'k_sim': [],
                        'positions': []})
        self.results = results
        self.cat = cat
        self.coyote = []
        for r in self.runners:
            self.coyote.append([r.data[k].channels for k in range(len(r.data))])

    def run(self, n_sim, plot = False):
        """ Generates simulated multi-resolution scenes and runs scarlet on them on-the-fly

        Parameters
        ----------
        nsim: int
            Number of simulations to generate
        plot: Bool
            If set to true, plots the result from scarlet: Convergence, model and residuals.
        """

        for i in range(n_sim):
            ns = np.int(np.random.rand(1)*(self.ngal-1)+1)
            pic_hr, pic_lr = gct.mk_scene(self.hr_dict,
                                      self.lr_dict,
                                      self.cat,
                                      (self.n_hr,self.n_hr),
                                      (self.n_lr,self.n_lr),
                                      ns, gal_type = 'real')
            shifts = pic_hr.shifts
            wcs_hr = pic_hr.wcs
            wcs_lr = pic_lr.wcs

            hr = pic_hr.cube
            lr = pic_lr.cube

            gs_hr = self.unconvolved
            gs_lr = self.unconvolved

            psf_hr = pic_hr.psf
            psf_lr = pic_lr.psf

            # Get the source coordinates from the HST catalog
            ytrue, xtrue = shifts[:,0], shifts[:,1]

            # Convert the HST coordinates to the HSC WCS
            ratrue, dectrue = wcs_hr.wcs_pix2world(ytrue,xtrue,0)
            catalog_true = np.array([ratrue, dectrue]).T

            hr = hr[None, :,:]
            data_hr =  Data(hr, wcs_hr, psf_hr, self.hr_dict['channels'])
            data_lr =  Data(lr, wcs_lr, psf_lr, self.lr_dict['channels'])

            for i,r in enumerate(self.runners):

                    if r.resolution == 'multi':
                        r.data = [data_lr, data_hr]
                        self.results[i]['resolution'] = 'Joint processing'
                    elif r.resolution == 'single':
                        if r.observations[0].frame.shape == hr.shape:
                            r.data = [data_hr]
                            self.results[i]['resolution'] = 'High resolution'
                        elif r.observations[0].frame.shape == lr.shape:
                            r.data = [data_lr]
                            self.results[i]['resolution'] = 'Low resolution'
                    if self.cats[i]:
                        r.initialize_sources(ks, catalog_true)
                    else:
                        r.initialize_sources(ks)

                    ############RUNNING things#############
                    r.run(it = 200, e_rel = 1.e-7, plot = plot)

                    model = r.blend.get_model()

                    model_psf = r.frame._psfs.image[0]
                    if self.results[i]['resolution'] == 'Joint processing':
                        render = [r.observations[0].render(model),
                                r.observations[1].render(model)]
                        truth = gs_hr
                        true_seds = [np.concatenate([seds_lr[i],
                                                     seds_hr[i]]) for i in range(ns)]
                    elif self.results[i]['resolution'] == 'High resolution':
                        render = [r.observations[0].render(model)]
                        truth = gs_hr
                        true_seds = seds_hr
                    elif self.results[i]['resolution'] == 'Low resolution':
                        render = [r.observations[0].render(model)]
                        truth = gs_lr
                        true_seds = seds_lr

                    sdrs = []
                    sed_sdrs = []
                    ndetect = len(r.ra_dec)
                    for k in range(ndetect):
                        true_source = scp.fftconvolve(truth[k], model_psf, mode = 'same')
                        source = r.sources[k].get_model(frame=r.observations[-1].frame)[0]
                        source=source / np.float(np.max(source)) * np.max(true_source)
                        spectrum = r.sources[k].get_model().sum(axis=(1, 2))

                        plt.figure(figsize = (30,10))
                        plt.subplot(131)
                        plt.imshow(source)
                        plt.colorbar()
                        plt.subplot(132)
                        plt.imshow(true_source)
                        plt.colorbar()
                        plt.subplot(133)
                        plt.imshow(source-true_source)
                        plt.colorbar()
                        plt.show()
                        plt.plot(np.array(true_seds[k]), 'or')
                        plt.plot(np.array(spectrum), 'ob')
                        plt.show()
                        plt.plot(np.array(true_seds[k])/np.array(spectrum), 'or')
                        plt.show()
                        sed_sdrs.append(SDR(np.array(true_seds)[k]/np.sum(true_source),
                                            np.array(spectrum)))
                        sdrs.append(SDR(true_source, source))

                    chis = []
                    for j,d in enumerate(r.data):
                        chis.append(chi(d.images,render[j]))
                    self.results[i]['chi'].append(chis)
                    self.results[i]['SDRs'].append(sdrs)
                    self.results[i]['SED_SDRs'].append(sed_sdrs)
                    self.results[i]['n_sources'].append(ns)
                    self.results[i]['k_sim'].append(ks)
                    self.results[i]['positions'].append(shifts)


    def plot(self):
        #Plot chi results
        plt.figure(0, figsize = (16,12))
        plt.title('$\chi^2$ per band', fontsize = 40)
        for i,res in enumerate(self.results):
            for j,c in enumerate(self.coyote[i]):
                if res['resolution'] == 'Low resolution':
                    label = 'Single resolution'
                    color = 'ob'
                    shift = 0.1
                elif res['resolution'] == 'High resolution':
                    label = None
                    color = 'ob'
                    shift = 0.1
                elif res['resolution'] == 'Joint processing':
                    label = 'Joint processing'
                    color = 'or'
                    shift = -0.1
                mean_chi = np.nanmean(np.array([chi[j] for chi in res['chi']]), axis = 0)
                std_chi = np.nanstd(np.array([chi[j] for chi in res['chi']]), axis = 0)
                if c == ['VIS']:

                    plt.errorbar(np.arange(len(c))+shift+6,
                            mean_chi,
                            yerr = std_chi,
                            fmt = color,
                            ms = 7,
                            elinewidth=3)
                else:
                    plt.errorbar(np.arange(len(c))+shift,
                            mean_chi,
                            yerr = std_chi,
                            fmt = color,
                            label = label,
                            ms = 7,
                            elinewidth=3)

        plt.xticks(ticks = np.arange(len(self.coyote[0][0] + self.coyote[1][0])),
                   labels = self.coyote[0][0] + self.coyote[1][0],
                   fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.ylabel('mean $\chi^2$', fontsize = 30)
        plt.xlabel('bands', fontsize = 30)
        plt.legend(fontsize = 25)
        plt.savefig('Chi2.png')
        plt.show()


        #SDR as a function of sources # per patch
        plt.figure(5, figsize = (16,12))
        plt.title('SDR$(n_{gal})$', fontsize = 40)
        for i in range(self.ngal):
            loc = np.where(np.array(self.results[0]['n_sources']) == i)
            if len(loc[0]) > 0:
                for j, res in enumerate(self.results):
                    sdr = np.nanmean(np.concatenate([res['SDRs'][int(l)] for l in loc[0]]))
                    std_sdr = np.nanstd(np.concatenate([res['SDRs'][int(l)] for l in loc[0]]))
                    if res['resolution'] == 'Low resolution':
                        color = '--og'
                        shift = -0.1
                    elif res['resolution'] == 'High resolution':
                        color = '--ob'
                        shift = 0.1
                    elif res['resolution'] == 'Joint processing':
                        color = '--or'
                        shift = 0
                    if i == 2:
                        plt.errorbar(i+shift,
                                sdr,
                                yerr = std_sdr,
                                fmt = color,
                                label = res['resolution'],
                                ms = 7,
                                elinewidth=3)
                    else:
                        plt.errorbar(i+shift,
                                sdr,
                                yerr = std_sdr,
                                fmt = color,
                                ms = 7,
                                elinewidth=3)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.ylabel('SDR', fontsize = 30)
        plt.xlabel('# sources per patch', fontsize = 30)
        plt.legend(fontsize = 25)
        plt.savefig('SDR(n).png')
        plt.show()


        #Chi as a function of #sources per patch
        plt.figure(2, figsize = (16,12))
        plt.title('$\chi^2(n_{gal})$', fontsize = 40)
        for i in range(self.ngal):
            loc = np.where(np.array(self.results[0]['n_sources']) == i)
            if len(loc[0]) > 0:
                for j, res in enumerate(self.results):

                    if res['resolution'] == 'Low resolution':
                        chi = np.nanmean(np.concatenate([res['chi'][int(l)] for l in loc[0]]))
                        std_chi = np.nanstd(np.concatenate([res['chi'][int(l)] for l in loc[0]]))
                        if i == 2:
                            plt.errorbar(i-0.15,
                                    chi,
                                    yerr = std_chi,
                                    fmt = '--sg',
                                    label = res['resolution'],
                                    ms = 7,
                                    elinewidth=3)
                        else:
                            plt.errorbar(i-0.15,
                                    chi,
                                    yerr = std_chi,
                                    fmt = '--sg',
                                    ms = 7,
                                    elinewidth=3)
                    elif res['resolution'] == 'High resolution':
                        chi = np.nanmean(np.concatenate([res['chi'][int(l)] for l in loc[0]]))
                        std_chi = np.nanstd(np.concatenate([res['chi'][int(l)] for l in loc[0]]))
                        if i == 2:
                            plt.errorbar(i+0.15,
                                    chi,
                                    yerr = std_chi,
                                    fmt = '--ob',
                                    label = res['resolution'],
                                    ms = 7,
                                    elinewidth=3)
                        else:
                            plt.errorbar(i+0.15,
                                    chi,
                                    yerr = std_chi,
                                    fmt = '--ob',
                                    ms = 7,
                                    elinewidth=3)

                    elif res['resolution'] == 'Joint processing':
                        chi_lr = np.nanmean(np.concatenate([res['chi'][int(l)][0] for l in loc[0]]))
                        chi_hr = np.nanmean(np.concatenate([res['chi'][int(l)][1] for l in loc[0]]))
                        std_chi_lr = np.nanstd(np.concatenate([res['chi'][int(l)][0] for l in loc[0]]))
                        std_chi_hr = np.nanstd(np.concatenate([res['chi'][int(l)][1] for l in loc[0]]))
                        if i == 2:
                            plt.errorbar(i+0.05,
                                    chi_hr,
                                    yerr = std_chi_hr,
                                    fmt = '--or',
                                    label = 'Joint hr',
                                    ms = 7,
                                    elinewidth=3,
                                    linewidth=3)
                        else:
                            plt.errorbar(i+0.05,
                                    chi_hr,
                                    yerr = std_chi_hr,
                                    fmt = '--or',
                                    ms = 7,
                                    elinewidth=3)
                        if i == 2:
                            plt.errorbar(i-0.05,
                                    chi_lr,
                                    yerr = std_chi_lr,
                                    fmt = '--sm',
                                    label = 'Joint lr',
                                    ms = 7,
                                    elinewidth=3)
                        else:
                            plt.errorbar(i-0.05,
                                    chi_lr,
                                    yerr = std_chi_lr,
                                    fmt = '--sm',
                                    ms = 7,
                                    elinewidth=3)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.ylabel('$\chi^2$', fontsize = 30)
        plt.xlabel('$n_gal$', fontsize = 30)
        plt.legend(fontsize = 25)
        plt.savefig('Chi2(n).png')
        plt.show()


        #SDR of galaxies
        plt.figure(4, figsize = (16,12))
        plt.title('Average SDR', fontsize = 40)
        for j, res in enumerate(self.results):
            isgal = [(ks != 'point') for ks in np.concatenate(res['k_sim'])]
            sdr_gal = np.nanmean(np.array(np.concatenate(res['SDRs']))[isgal])
            std_sdr_gal = np.nanstd(np.array(np.concatenate(res['SDRs']))[isgal])

            plt.errorbar(j,
                         sdr_gal,
                         yerr = std_sdr_gal,
                         fmt = 'o',
                         label = res['resolution'],
                         ms = 7,
                         elinewidth=3)
        plt.yticks(fontsize = 25)
        plt.xticks(ticks = np.arange(len(self.results)), labels = [res['resolution'] for res in self.results], fontsize = 25)
        plt.ylabel('SDR', fontsize = 30)
        #plt.legend()
        plt.savefig('SDR.png')
        plt.show()

        #SDR of galaxy spectra
        plt.figure(4, figsize = (16,12))
        plt.title('Spectrum SDR', fontsize = 40)
        for j, res in enumerate(self.results):
            isgal = [(ks != 'point') for ks in np.concatenate(res['k_sim'])]
            sdr = np.nanmean(np.array(np.concatenate(res['SED_SDRs']))[isgal])
            std_sdr = np.nanstd(np.array(np.concatenate(res['SED_SDRs']))[isgal])

            plt.errorbar(j,
                         sdr,
                         yerr = std_sdr,
                         fmt = 'o',
                         label = res['resolution'],
                         ms = 7,
                         elinewidth=3)
        plt.yticks(fontsize = 25)
        plt.xticks(ticks = np.arange(len(self.results)), labels = [res['resolution'] for res in self.results], fontsize = 25)
        plt.ylabel('Spectrum SDR', fontsize = 30)
        #plt.legend()
        plt.savefig('SED_SDR.png')
        plt.show()

         #SDR of star spectra
        plt.figure(5, figsize = (16,12))
        plt.title('Point Source Spectrum SDR', fontsize = 40)
        for j, res in enumerate(self.results):
            isgal = [(ks != 'point') for ks in np.concatenate(res['k_sim'])]
            sdr = np.nanmean(np.array(np.concatenate(res['SED_SDRs']))[not isgal])
            std_sdr = np.nanstd(np.array(np.concatenate(res['SED_SDRs']))[not isgal])

            plt.errorbar(j,
                         sdr,
                         yerr = std_sdr,
                         fmt = 'o',
                         label = res['resolution'],
                         ms = 7,
                         elinewidth=3)
        plt.yticks(fontsize = 25)
        plt.xticks(ticks = np.arange(len(self.results)), labels = [res['resolution'] for res in self.results], fontsize = 25)
        plt.ylabel('Spectrum SDR', fontsize = 30)
        #plt.legend()
        plt.savefig('Point_SED_SDR.png')
        plt.show()

        #SDR of spectrum as a function of sources # per patch
        plt.figure(6, figsize = (16,12))
        plt.title('Spectrum SDR$(n_{gal})$', fontsize = 40)
        for i in range(self.ngal):
            loc = np.where(np.array(self.results[0]['n_sources']) == i)
            if len(loc[0]) > 0:
                for j, res in enumerate(self.results):
                    sdr = np.nanmean(np.concatenate([res['SED_SDRs'][int(l)] for l in loc[0]]))
                    std_sdr = np.nanstd(np.concatenate([res['SED_SDRs'][int(l)] for l in loc[0]]))
                    if res['resolution'] == 'Low resolution':
                        color = '--og'
                        shift = -0.1
                    elif res['resolution'] == 'High resolution':
                        color = '--ob'
                        shift = 0.1
                    elif res['resolution'] == 'Joint processing':
                        color = '--or'
                        shift = 0
                    if i == 2:
                        plt.errorbar(i+shift,
                                sdr,
                                yerr = std_sdr,
                                fmt = color,
                                label = res['resolution'],
                                ms = 7,
                                elinewidth=3)
                    else:
                        plt.errorbar(i+shift,
                                sdr,
                                yerr = std_sdr,
                                fmt = color,
                                ms = 7,
                                elinewidth=3)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.ylabel('Spectrum SDR', fontsize = 30)
        plt.xlabel('# sources per patch', fontsize = 30)
        plt.legend(fontsize = 25)
        plt.savefig('SED_SDR(n).png')
        plt.show()

        pass