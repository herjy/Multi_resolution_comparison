
import numpy as np
import scipy.signal as scp
import scipy.stats as stats
import scarlet
from scarlet_extensions.initialization.detection import Data
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
    sigma_ROMAN = 0.11*np.array([1.68, 1.69, 1.86, 2.12, 2.44, 2.71]) #https://arxiv.org/pdf/1702.01747.pdf Z-band
    sigma_RUBIN = np.array([0.327, 0.31, 0.297, 0.285, 0.276, 0.267]) #https://www.lsst.org/about/camera/features
    sigma_EUCLID = np.array([0.16]) #https://sci.esa.int/documents/33859/36320/1567253682555-Euclid_presentation_Paris_1Dec2009.pdf
    sigma_HST = np.array([0.074]) #Source https://hst-docs.stsci.edu/display/WFC3IHB/6.6+UVIS+Optical+Performance#id-6.6UVISOpticalPerformance-6.6.1 800nm
    sigma_HSC = np.array([0.306, 0.285, 0.238, 0.268, 0.272]) #https://hsc-release.mtk.nao.ac.jp/doc/ deep+udeep


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
           'sky':np.array([22]),
           'exp_time': np.array([3000]),
           'zero_point': np.array([20])}
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
             'channels': ['F062', 'Z087', 'Y106', 'J129', 'H158', 'F184'],
             'sky':np.array([22, 22, 22, 22, 22, 22]), ## Not Checked!!!
             'exp_time': np.array([3000,3000,3000,3000,3000,3000]),## Not Checked!!!
             'zero_point': np.array([26.99, 26.39, 26.41, 26.35, 26.41, 25.96])}
    RUBIN = {'name': 'RUBIN',
             'pixel': pix_RUBIN,
             'psf': sigma_RUBIN,
             'channels': ['u','g','r','i','z','y'],
             'sky': np.array([22.9, 22.3, 21.2, 20.5, 19.6, 18.6]),
             'exp_time': np.array([1680, 2400, 5520, 5520, 4800, 4800]),
             'zero_point': np.array([9.16, 50.70, 43.70, 32.36, 22.68, 10.58])}

    return HST, EUCLID, ROMAN, HSC, RUBIN

HST, EUCLID, ROMAN, HSC, RUBIN = load_surveys()

def SDR(X_true, X):
    """Source distortion ratio between an expected value and its estimate. The higher the SDR the better X_true and X agree"""
    return 10*np.log10(np.sum(X_true**2)**0.5/np.sum((X_true-X)**2)**0.5)

def chi(image, model):
    return image.shape[0]/image.size*(np.sum((image - model)**2, axis = (-2,-1))/scarlet.wavelet.mad_wavelet(image)**2)


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
                        'positions': [],
                        'mags': []})
        self.results = results
        self.cat = cat
        self.coyote = []
        for r in self.runners:
            self.coyote.append([r.data[k].channels for k in range(len(r.data))])

    def run(self, n_sim, plot = False, norms = None, init_param = True):
        """ Generates simulated multi-resolution scenes and runs scarlet on them on-the-fly

        Parameters
        ----------
        nsim: int
            Number of simulations to generate
        plot: Bool
            If set to true, plots the result from scarlet: Convergence, model and residuals.
        init_param: Bool
            If set to true, the initialisation uses galsim's parametric fits to the simulated galaxy profiles.
        """

        for i in range(n_sim):
            ns = np.int(np.random.rand(1)*(self.ngal-1)+1)
            pic_hr, pic_lr = gct.mk_scene(self.hr_dict,
                                          self.lr_dict,
                                          self.cat,
                                          (self.n_hr,self.n_hr),
                                          (self.n_lr,self.n_lr),
                                          ns,
                                          gal_type = 'real',
                                          pt_fraction = 0)
            shifts = np.array(pic_hr.shifts)
            wcs_hr = pic_hr.wcs
            wcs_lr = pic_lr.wcs

            hr = pic_hr.cube
            lr = pic_lr.cube

            gs_hr = pic_hr.galaxies
            gs_lr = pic_lr.galaxies

            psf_hr = pic_hr.psfs
            psf_lr = pic_lr.psfs

            ks = pic_hr.ks

            seds_hr = pic_hr.seds
            seds_lr = pic_lr.seds

            mags_hr = pic_hr.mags
            mags_lr = pic_lr.mags

            # Get the source coordinates from the HST catalog
            ytrue, xtrue = shifts[:,0], shifts[:,1]

            # Convert the HST coordinates to the HSC WCS
            ratrue, dectrue = wcs_hr.wcs_pix2world(ytrue,xtrue,0)

            catalog_true = np.array([ratrue, dectrue]).T

            data_hr = Data(hr, wcs_hr, psf_hr, self.hr_dict['channels'])
            data_lr = Data(lr, wcs_lr, psf_lr, self.lr_dict['channels'])

            for i,r in enumerate(self.runners):

                    if r.resolution == 'multi':
                        r.data = [data_lr, data_hr]
                        self.results[i]['resolution'] = 'Joint processing'
                    elif r.resolution == 'single':

                        if r.observations[0].data.shape == hr.shape:
                            r.data = [data_hr]
                            self.results[i]['resolution'] = 'High resolution'
                        elif r.observations[0].data.shape == lr.shape:
                            r.data = [data_lr]
                            self.results[i]['resolution'] = 'Low resolution'
                    if init_param == False:
                        if self.cats[i]:
                            r.initialize_sources(ks, catalog_true)
                        else:
                            r.initialize_sources(ks)
                    else:
                        if self.cats[i]:
                            if self.results[i]['resolution'] == 'Joint processing':
                                r.initialize_sources(ks, catalog_true, morph=pic_hr.parametrics)
                            elif self.results[i]['resolution'] == 'High resolution':
                                r.initialize_sources(ks, catalog_true, morph=pic_hr.parametrics)
                            elif self.results[i]['resolution'] == 'Low resolution':
                                r.initialize_sources(ks, catalog_true, morph=pic_lr.parametrics)
                        else:
                            r.initialize_sources(ks)
                    ############RUNNING things#############
                    if norms is not None:
                        norm = norms[1]
                    else:
                        norm = None
                    r.run(it = 200, e_rel = 1.e-7, plot = plot, norms = norm)

                    model = r.blend.get_model()

                    model_psf = r.frame._psf.get_model()[0]
                    if self.results[i]['resolution'] == 'Joint processing':
                        render = [r.observations[0].render(model),
                                r.observations[1].render(model)]
                        truth = gs_hr
                        true_seds = [np.concatenate([seds_lr[i],
                                                     seds_hr[i]]) for i in range(ns)]
                        mags = [np.concatenate([mags_lr[i],
                                                mags_hr[i]]) for i in range(ns)]
                    elif self.results[i]['resolution'] == 'High resolution':
                        render = [r.observations[0].render(model)]
                        truth = gs_hr
                        true_seds = seds_hr
                        mags = mags_hr
                    elif self.results[i]['resolution'] == 'Low resolution':
                        render = [r.observations[0].render(model)]
                        truth = gs_lr
                        true_seds = seds_lr
                        mags = mags_lr

                    true_seds = np.array(true_seds)
                    sdrs = []
                    sed_sdrs = []

                    ndetect = len(r.ra_dec)
                    obs = r.observations[-1]
                    for k in range(ndetect):
                        true_source = np.zeros(r.frame.shape, dtype=r.frame.dtype)
                        source = r.sources[k].get_model(frame=r.frame)[0]
                        data_slice, model_slice = obs.renderer.slices
                        obs.renderer.map_channels(true_source)[model_slice] = \
                            (np.ones(obs.shape[0])[:, None, None] * truth[k][None, :, :])[data_slice]

                        true_source = scp.fftconvolve(true_source[-1], model_psf, mode = 'same')

                        source = source / np.float(np.sum(source))
                        spectrum = np.concatenate([r.sources[k].get_model(frame=obs).sum(axis=(1, 2)) for obs in r.observations])

                        sed_sdrs.append(SDR(np.array(true_seds)[k],
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
                    self.results[i]['mags'].append(mags)


    def plot(self, spectrum = False, mags = True):
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
                mean_chi = np.nanmedian(np.array([chi[j] for chi in res['chi']]), axis = 0)
                std_chi = stats.median_absolute_deviation(np.array([chi[j] for chi in res['chi']]), axis = 0, nan_policy = 'omit')
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
        plt.ylabel('median $\chi^2$', fontsize = 30)
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
                    sdr = np.nanmedian(np.concatenate([res['SDRs'][int(l)] for l in loc[0]]))
                    std_sdr = stats.median_absolute_deviation(np.concatenate([res['SDRs'][int(l)] for l in loc[0]]), nan_policy = "omit")
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
                        chis = np.concatenate([res['chi'][int(l)] for l in loc[0]])
                        chi = np.nanmedian(chis)
                        std_chi = stats.median_absolute_deviation(chis, axis = None, nan_policy = "omit")

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
                        chi = np.nanmedian(np.concatenate([res['chi'][int(l)] for l in loc[0]]))
                        std_chi = stats.median_absolute_deviation(np.concatenate([res['chi'][int(l)] for l in loc[0]]), nan_policy = "omit")
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
                        chi_lr = np.nanmedian(np.concatenate([res['chi'][int(l)][0] for l in loc[0]]))
                        chi_hr = np.nanmedian(np.concatenate([res['chi'][int(l)][1] for l in loc[0]]))
                        std_chi_lr = stats.median_absolute_deviation(np.concatenate([res['chi'][int(l)][0] for l in loc[0]]), nan_policy = "omit")
                        std_chi_hr = stats.median_absolute_deviation(np.concatenate([res['chi'][int(l)][1] for l in loc[0]]), nan_policy = "omit")
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

        if mags:
            plt.figure(figsize=(16,12))
            plt.title('SDR per VIS mag bin', fontsize=40)
            vmags = np.concatenate(self.results[1]['mags'])[:,0]
            min_mags = 20
            max_mags = 27
            xmags = np.linspace(min_mags, max_mags, max_mags-min_mags+1)
            bin_size = xmags[1]-xmags[0]

            for kr,r in enumerate(self.results):
                if r['resolution'] == 'Low resolution':
                    color = 'og'
                    shift = -0.02
                    alpha = 0.2
                elif r['resolution'] == 'High resolution':
                    color = 'ob'
                    shift = 0.02
                    alpha = 1
                elif r['resolution'] == 'Joint processing':
                    color = 'or'
                    shift = 0
                    alpha = 1
                sdrs = np.concatenate(r['SDRs'])
                binned_mag = []
                std_mag = []
                for b in xmags:

                    binned_mag.append(np.mean(sdrs[np.abs(vmags-b-0.5) < bin_size/2.]))
                    std_mag.append(np.std(sdrs[np.abs(vmags-b-0.5) < bin_size/2.]))

                plt.errorbar(xmags+shift+0.5,
                             binned_mag,
                             xerr=0.5,
                             yerr = std_mag,
                             label = r['resolution'],
                             ms = 7,
                             capsize = 3,
                             fmt = '--'+color,
                             elinewidth=3,
                             alpha = alpha)
                #plt.plot(mags, sdrs, color, label=r['resolution'], alpha = 0.2)
                plt.xlabel('VIS magnitude', fontsize=30)
                plt.ylabel('SDR', fontsize=30)
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                plt.legend(fontsize=25)
            plt.savefig('SDR(Euclid_mag).png')
            plt.show()

            plt.figure(figsize=(16,12))
            plt.title('SDR per min Rubin mag bin', fontsize=40)
            rmags = np.min(np.concatenate(self.results[0]['mags']), axis = -1)

            for kr,r in enumerate(self.results):
                if r['resolution'] == 'Low resolution':
                    color = 'og'
                    shift = -0.02
                    alpha = 1
                elif r['resolution'] == 'High resolution':
                    color = 'ob'
                    shift = 0.02
                    alpha = 0.2
                elif r['resolution'] == 'Joint processing':
                    color = 'or'
                    shift = 0
                    alpha = 1
                sdrs = np.concatenate(r['SDRs'])
                binned_mag = []
                std_mag = []
                for b in xmags:

                    binned_mag.append(np.mean(sdrs[np.abs(rmags-b-0.5) < bin_size/2.]))
                    std_mag.append(np.std(sdrs[np.abs(rmags-b-0.5) < bin_size/2.]))

                plt.errorbar(xmags+shift+0.5,
                             binned_mag,
                             xerr=0.5,
                             yerr=std_mag,
                             label=r['resolution'],
                             ms=7,
                             capsize=3,
                             fmt='--'+color,
                             elinewidth=3,
                             alpha=alpha)

                plt.xlabel('min Rubin magnitude', fontsize=30)
                plt.ylabel('SDR', fontsize=30)
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                plt.legend(fontsize=25)
            plt.savefig('SDR(Rubin_mag).png')
            plt.show()

            #VIS mag vs r-mag
            plt.figure(figsize=(70,20))
            plt.suptitle('SDR per magnitude bin', fontsize=90)

            sdr_tab = []
            sdr_std_tab =[]
            for kr,r in enumerate(self.results):
                if r['resolution'] == 'Low resolution':
                    color = 'og'
                    shift = -0.02
                    plt.subplot(131)
                elif r['resolution'] == 'High resolution':
                    color = 'ob'
                    shift = 0.02
                    plt.subplot(132)
                elif r['resolution'] == 'Joint processing':
                    color = 'or'
                    shift = 0
                    plt.subplot(133)
                sdrs = np.concatenate(r['SDRs'])
                binned_mag = np.zeros((len(xmags), len(xmags)))
                counts = np.zeros((len(xmags), len(xmags)))
                std_mag = np.zeros((len(xmags), len(xmags)))
                for ib,b in enumerate(xmags):
                    vcond = np.abs(vmags-b-0.5) < bin_size/2.
                    for ic, c in enumerate(xmags):
                        rcond = np.abs(rmags-c-0.5) < bin_size/2.
                        binned_mag[ib,ic] = (np.mean(sdrs[vcond*rcond]))
                        std_mag[ib,ic] = (np.std(sdrs[vcond*rcond]))
                        counts[ib,ic] = np.size(sdrs[vcond*rcond])


                plt.title(r['resolution'], fontsize = 70)
                plt.imshow(binned_mag, vmax = 10, vmin = -5, cmap = 'gnuplot')
                cbar = plt.colorbar(shrink = 0.83)
                cbar.ax.tick_params(labelsize=45)
                plt.ylim((-0.5, max_mags-min_mags+.5))
                plt.xlim((-0.5, max_mags-min_mags+.5))
                plt.ylabel('VIS mag', fontsize = 55)
                plt.xlabel('Min Rubin mag', fontsize = 55)
                plt.xticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize = 45)
                plt.yticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize = 45)

                sdr_tab.append(binned_mag)
                sdr_std_tab.append(std_mag)
            plt.savefig('SDR_per_mag.png')
            plt.show()

            amp = np.nanmax(np.abs(sdr_tab[0]-sdr_tab[-1]))
            plt.figure(figsize=(50, 20))
            plt.subplot(121)
            plt.title('Rubin vs Joint resolution', fontsize = 70)
            plt.imshow(sdr_tab[-1]-sdr_tab[0], vmin = -amp, vmax = amp, cmap='seismic')
            cbar = plt.colorbar(shrink=0.93)
            cbar.ax.tick_params(labelsize=45)
            plt.ylim((-0.5, max_mags-min_mags+.5))
            plt.xlim((-0.5, max_mags-min_mags+.5))
            plt.ylabel('VIS mag', fontsize=55)
            plt.xlabel('Min Rubin mag', fontsize=55)
            plt.xticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize=45)
            plt.yticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize=45)

            amp = np.nanmax(np.abs(sdr_tab[1] - sdr_tab[-1]))
            plt.subplot(122)
            plt.title('Euclid vs Joint resolution', fontsize = 70)
            plt.imshow(sdr_tab[-1] - sdr_tab[1], vmin = -amp, vmax = amp, cmap='seismic')
            cbar = plt.colorbar(shrink=0.93)
            cbar.ax.tick_params(labelsize=45)
            plt.ylim((-0.5, max_mags-min_mags+.5))
            plt.xlim((-0.5, max_mags-min_mags+.5))
            plt.ylabel('VIS mag', fontsize=55)
            plt.xlabel('Min Rubin mag', fontsize=55)
            plt.xticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize=45)
            plt.yticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize=45)
            plt.savefig('Single_vs_joint.png')
            plt.show()

            # Average sdr along the diagonal
            diags=[]
            diags_std = []
            for indi, sdr_array in enumerate(sdr_tab):
                n1,n2 = np.shape(sdr_array)
                diags.append(sdr_array[np.arange(n1).astype(int),np.arange(n2).astype(int)])
                diags_std.append(sdr_std_tab[indi][np.arange(n1).astype(int), np.arange(n2).astype(int)])



            plt.figure(figsize = (16,12))
            plt.suptitle('SDRs at equal magnitudes', fontsize = 50)

            plt.errorbar(xmags - 0.02+0.5,
                         diags[0],
                         xerr=0.5,
                         yerr=diags_std[0],
                         fmt='--og',
                         ms=7,
                         capsize=3,
                         elinewidth=3,
                         label='Rubin min mag')
            plt.errorbar(xmags+0.5,
                         diags[1],
                         xerr=0.5,
                         yerr=diags_std[1],
                         fmt='--ob',
                         ms=7,
                         capsize=3,
                         elinewidth=3,
                         label='Euclid VIS mag')
            plt.errorbar(xmags + 0.02+0.5,
                         diags[2],
                         xerr=0.5,
                         yerr=diags_std[2],
                         fmt='--or',
                         ms=7,
                         capsize=3,
                         elinewidth=3,
                         label='Joint resolution')
            plt.xlabel('Rubin mag = VIS mag', fontsize=30)
            plt.ylabel('SDR', fontsize=30)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.legend(fontsize=25)
            plt.savefig('SDR_equal_mag.png')
            plt.show()

            print(np.min(counts))
            plt.figure(figsize=(30, 20))
            plt.figure(figsize=(50, 20))
            plt.subplot(121)
            plt.title('Galaxy mag distribution', fontsize = 70)
            plt.imshow(counts, cmap='gnuplot')
            cbar = plt.colorbar(shrink=0.93)
            cbar.ax.tick_params(labelsize=45)
            plt.ylim((-0.5, max_mags-min_mags+.5))
            plt.xlim((-0.5, max_mags-min_mags+.5))
            plt.ylabel('VIS mag', fontsize=55)
            plt.xlabel('Min Rubin mag', fontsize=55)
            plt.xticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize=45)
            plt.yticks(np.arange(len(xmags))[::2], np.round(xmags, 2)[::2]+0.5, fontsize=45)
            plt.savefig('mag_distribution.png')
            plt.show()

        if spectrum:
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