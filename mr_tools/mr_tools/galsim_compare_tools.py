import galsim
from astropy import wcs as WCS
import numpy as np
import scipy.signal as scp
import scarlet
from scarlet_extensions.initialization.detection import Data
import pickle
import matplotlib.pyplot as plt

center_ra = 19.3*galsim.degrees     # The RA, Dec of the center of the image on the sky
center_dec = -33.1*galsim.degrees
def load_surveys():
    """Creates dictionaries for the HST, EUCLID, WFRIST, HCS anf LSST surveys
    that contain their names, pixel sizes and psf fwhm in arcseconds"""
    pix_ROMAN = 0.11
    pix_RUBIN = 0.2
    pix_HST = 0.06
    pix_EUCLID = 0.068
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
           'channels': ['f814w']}
    HSC = {'name': 'HSC', 
           'pixel': pix_HSC,
           'psf': sigma_HSC, 
           'channels': ['g','r','i','z','y'],
           'sky': np.array([21.4, 20.6, 19.7, 18.3, 17.9]),
           'exp_time': np.array([600, 600, 1200, 1200, 1200]).
          ,'zero_point': np.array([91.11, 87.74, 69.80, 29.56, 21.53])}
    ROMAN = {'name': 'ROMAN', 
             'pixel': pix_ROMAN,
             'psf': sigma_ROMAN, 
             'channels': ['R062', 'Z087', 'Y106', 'J129', 'H158', 'F184']}
    RUBIN = {'name': 'RUBIN', 
             'pixel': pix_RUBIN,
             'psf': sigma_RUBIN, 
             'channels': ['u','g','r','i','z','y'], 
             'sky': np.array([22.9, 22.3, 21.2, 20.5, 19.6, 18.6]),
             'exp_time': np.array([1680, 2400, 5520, 5520, 4800, 4800]),
             'zero_point': np.array([9.16, 50.70, 43.70, 32.36, 22.68, 10.58])}
    
    return HST, EUCLID, ROMAN, HSC, RUBIN

HST, EUCLID, ROMAN, HSC, RUBIN = load_surveys()

def mk_wcs(theta, pix, center, shape, naxis = 2):
    '''Creates wcs for an image
    
    Parameters
    ----------
    theta: float
        rotation angle for the image
    pix: float
        pixel size in arcseconds
    center: tuple
        position of the reference pixel used as the center of the affin transform for the wcs
    shape: tuple
        shape of the image
    
    Returns
    -------
    wcs: WCS
    '''
    #Affine transformation
    dudx = np.cos(theta) * pix
    if theta == 0:
        dudy = 0
        dvdx = 0
    else:
        dudy = -np.sin(theta) * pix
        dvdx = np.sin(theta) * pix
    dvdy = np.cos(theta) * pix
    
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=center)
    #Image center
    sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)
    
    #Creating WCS
    w = WCS.WCS(naxis=2)
    galsim_wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.crpix = galsim_wcs.crpix
    w.wcs.pc = galsim_wcs.cd
    w.wcs.crval = [galsim_wcs.center._ra._rad, galsim_wcs.center._dec._rad]
    w.array_shape = shape
    return w
    

def mk_sim(k, hr_dir, lr_dir, shape_hr, shape_lr, npsf, cat, shift = (0,0), gal_type = 'real', smooth = False, point = False):
    '''creates low and high resolution images of a galaxy profile with different psfs from the list of galaxies in the COSMOS catalog
    
    Parameters
    ----------
    k: int
        index of the galaxy to draw from the COSMOS catalog
    hr_dir: dictionary
        dictionary that contains the information for the high resolution survey
    lr_dir: dictionary
        dictionary that contains the information for the low resolution survey
    shape_hr: tuple of ints
        shape of the hr image
    shape_lr: tuple of ints
        shape of the lr image
    npsf: int
        size on-a-side of a psf (in pixels)
    cat: list
        catalog where to draw galaxies from
    gal_type: 'string'
        either use 'real' or 'parametric' light profiles.
    smooth: 'bool'
        set to true to trigger a convolution by a gaussian and smooth the noise out.
    point: 'bool'
        set to true to make the source a point source.
    Returns
    -------
    im_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    im_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    psf_hr: numpy array
        psf of the high resolution image
    psf_lr: numpy array
        psf of the low resolution image
    '''
    pix_hr = hr_dir['pixel']
    pix_lr = lr_dir['pixel']
    sigma_hr = hr_dir['psf']
    sigma_lr = lr_dir['psf']
 
    #Rotation angle
    theta = np.random.randn(1)*np.pi*0
    angle = galsim.Angle(theta,galsim.radians)
    
    #Image frames
    im_hr = galsim.Image(shape_hr[0], shape_hr[1], scale=pix_hr)
    im_lr = galsim.Image(shape_lr[0], shape_lr[1], scale=pix_lr)
    
    if point:
        gal = galsim.Gaussian(sigma = 0.02).withFlux(1)
    if not point:
        #Galaxy profile
        gal = cat.makeGalaxy(k, gal_type = gal_type, noise_pad_size=shape_lr[0] * pix_lr * 0).withFlux(1)
    
        if smooth == True:
            gal = galsim.Convolve(gal, galsim.Gaussian(sigma=pix_hr))
            
    gal = gal.shift(dx=shift[0], dy=shift[1])
    ## PSF is a Moffat profile dilated to the sigma of the corresponding survey
    psf_hr_int = galsim.Moffat(2, HST['pixel']).dilate(sigma_hr/HST['psf']).withFlux(1.)
    ## Draw PSF
    psf_hr = psf_hr_int.drawImage(nx=npsf,ny=npsf, method = 'real_space',
                                  use_true_center = True, scale = pix_hr).array
    ## Make sure PSF vanishes on the edges of a patch that has the shape of the initial npsf
    psf = psf_hr-psf_hr[0, int(npsf/2)]*2
    psf[psf<0] = 0
    psf_hr = psf/np.sum(psf)
    ## Interpolate the new 0-ed psf 
    psf_hr_int = galsim.InterpolatedImage(galsim.Image(psf_hr), scale = pix_hr).withFlux(1.)
    ## Re-draw it (with the correct fulx)
    psf_hr = psf_hr_int.drawImage(nx=npsf,ny=npsf, method = 'real_space',
                                  use_true_center = True, scale = pix_hr).array
    ## Same process with the low resolution PSF
    psf_lr_int = galsim.Moffat(2, HST['pixel']).dilate(sigma_lr/HST['psf']).withFlux(1.)
    # Draw it
    psf_lr = psf_lr_int.drawImage(nx=npsf,ny=npsf, method = 'real_space',
                                  use_true_center = True, scale = pix_lr).array
    ## Make sure it goes to 0
    psf = psf_lr-psf_lr[0, int(npsf/2)]*2
    psf[psf<0] = 0
    psf_lr = psf/np.sum(psf)
    ## Interpolate the 0-ed PSF 
    psf_lr_int = galsim.InterpolatedImage(galsim.Image(psf_lr), scale = pix_lr).withFlux(1.)
    ## Draw it with the right flux
    psf_lr = psf_lr_int.drawImage(nx=npsf,ny=npsf, method = 'real_space',
                                  use_true_center = True, scale = pix_lr).array

    # Convolve galaxy profile by PSF, rotate and sample at high resolution
    i_hr = gal.drawImage(nx=shape_hr[0],ny=shape_hr[1], 
                         use_true_center = True, method = 'no_pixel',
                         scale = pix_hr, dtype = np.float64)
    # Convolve galaxy profile by PSF, rotate and sample at low resolution
    i_lr = gal.rotate(angle).drawImage(nx=shape_lr[0],
                                       ny=shape_lr[1], 
                                       use_true_center = True, 
                                       method = 'no_pixel',
                                       scale = pix_lr, 
                                       dtype = np.float64)
    
    # Make WCSs
    wcs_hr = mk_wcs(0, pix_hr, galsim.PositionD(i_hr.true_center), shape_hr)
    wcs_lr = mk_wcs(0, pix_lr, galsim.PositionD(i_lr.true_center), shape_lr)

    # Convolve galaxy profile by PSF, rotate and sample at high resolution
    im_hr = galsim.Convolve(gal, psf_hr_int).drawImage(nx=shape_hr[0],ny=shape_hr[1], 
                                                       use_true_center = True, method = 'no_pixel',
                                                   scale = pix_hr, dtype = np.float64)
    # Convolve galaxy profile by PSF, rotate and sample at low resolution
    im_lr = galsim.Convolve(gal.rotate(angle), psf_lr_int).drawImage(nx=shape_lr[0],
                                                                     ny=shape_lr[1], 
                                                                     use_true_center = True, 
                                                                     method = 'no_pixel',
                                                                     scale = pix_lr, 
                                                                     dtype = np.float64)
   
    return im_hr.array, im_lr.array, wcs_hr, wcs_lr, psf_hr[None,:,:], psf_lr[None,:,:], theta, i_hr.array, i_lr.array
    
    
def mk_scene(hr_dict, lr_dict, cat, shape_hr, shape_lr, n_gal, gal_type):
    """ Generates blended scenes at two resolutions
    
    Parameters
    ----------
    hr_dict, lr_dict: dictionaries
        two dictionaries that contain the information for surveys (pixel scale, name and psf size)
    cat: catalog
        catalog of sources for galsim
    shape_hr, shape_lr: tuples
        2D shapes of the desired images for surveys indicated in the dictionaries
    ngal: int
        number of galaxies to draw on the scene
    gal_type: 'string'
        either use 'real' or 'parametric' light profiles.
    """
    pix_hr = hr_dict['pixel']
    
    lr = 0
    hr = 0
    loc = []
    ks = []
    gals_hr = []
    gals_lr = []
    seds_hr = []
    seds_lr = []
    
    for i in range(n_gal):
        
        k = np.int(np.random.rand(1)*len(cat))
        shift = ((np.random.rand(2)-0.5) * shape_hr * pix_hr * 2 / 3)
        
        sed_lr = (np.random.rand(len(lr_dict['channels']))*0.8)+0.2
        sed_hr = (np.random.rand(len(hr_dict['channels']))*0.8)+0.2
        
        toss = np.random.rand(1) < 0.1
        
        if lr_dict['name'] == 'RUBIN':
            lr_dict['psf'] = Rubin_psf[i]
        
        if toss:
            k = 'point'
            sed_lr *= 100
            sed_hr *= 100
        ks.append(k)
        ihr, ilr, wcs_hr, wcs_lr, phr, plr, _, i_hr, i_lr = mk_sim(k, hr_dict, lr_dict, 
                                       shape_hr, shape_lr, 41, cat, 
                                       shift = shift, gal_type = gal_type, smooth = True, point = toss)
        
        gals_hr.append(i_hr)
        gals_lr.append(i_lr)
        
        Ihr = ihr * sed_hr
        Ilr = ilr[None, :, :] * sed_lr[:, None, None]
        hr += ihr * sed_hr
        lr += ilr[None, :, :] * sed_lr[:, None, None]
        
        loc.append([shift[0]/pix_hr+(shape_hr[0]-1)/2,shift[1]/pix_hr+(shape_hr[1]-1)/2])
        seds_hr.append(np.sum(Ihr[None,:,:], axis = (-2,-1)))
        seds_lr.append(np.sum(Ilr, axis = (-2,-1)))
    
    lr += np.random.randn(*lr.shape) * np.sum(lr**2)**0.5/np.size(lr) * 100 / np.ones_like(sed_lr[:, None, None])
    hr += np.random.randn(*hr.shape) * np.sum(hr**2)**0.5/np.size(hr) * 150
    plr = plr * np.ones(len(lr_dict['channels']))[:, None, None]
    phr = phr * np.ones(len(hr_dict['channels']))[:, None, None]
    return hr, lr, wcs_hr, wcs_lr, phr, plr, np.array(loc), ks, gals_lr, gals_hr, seds_hr, seds_lr

def setup_scarlet(data_hr, data_lr, wcs_hr, wcs_lr, psf_hr, psf_lr, channels, coverage = 'union'):
    '''Performs the initialisation steps for scarlet to run its resampling scheme
    
    Prameters
    ---------
    data_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    data_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    psf_hr: numpy array
        psf of the high resolution image
    psf_lr: numpy array
        psf of the low resolution image
    channels: tuple
        names of the channels
    
    Returns
    -------
    obs: array of observations
        array of scarlet.Observation objects initialised for resampling
    '''
    #Extract data
    im_hr = data_hr[None, :, :]
    im_lr = data_lr[None, :, :]
    
    # define two observation objects and match to frame  
    obs_hr = scarlet.Observation(im_hr, wcs=wcs_hr, psfs=psf_hr, channels=[channels[1]])
    obs_lr = scarlet.Observation(im_lr, wcs=wcs_lr, psfs=psf_lr, channels=[channels[0]])

    # Keep the order of the observations consistent with the `channels` parameter
    # This implementation is a bit of a hack and will be refined in the future
    obs = [obs_lr, obs_hr]
    
    scarlet.Frame.from_observations(obs, obs_id = 1, coverage = coverage)
    return obs
    
def interp_galsim(data_hr, data_lr, diff_psf, angle, h_hr, h_lr):
    '''Apply resampling from galsim
    
    Prameters
    ---------
    data_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    data_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    diff_hr: numpy array
        difference kernel betwee the high and low resolution psf
    angle: float
        angle between high and low resolution images
    h_hr: float
        scale of the high resolution pixel (arcsec)
    h_lr: float
        scale of the low resolution pixel (arcsec)
        
    Returns
    -------
    interp_gal: galsim.Image
        image interpolated at low resolution
    '''
    # Load data
    im_hr = data_hr[None, :, :]
    im_lr = data_lr[None, :, :]
    _,n_hr,n_hr = im_hr.shape
    _,n_lr,n_lr = im_lr.shape
    
    # Interpolate hr image
    gal_hr = galsim.InterpolatedImage(galsim.Image(im_hr[0]), scale = h_hr)
    
    # Rotate hr galaxy to lr frame
    rot_gal = gal_hr.rotate(galsim.Angle(angle, galsim.radians))
    
    # Convolve hr galaxy by diff kernel at hr
    conv_gal = galsim.Convolve(rot_gal, diff_psf)
     
    # Downsamples to low resolution
    interp_gal = conv_gal.drawImage(nx=n_lr,ny=n_lr, scale=h_lr, method = 'no_pixel',)
    
    return interp_gal

def SDR(X_true, X):
    """Source distortion ratio between an expected value and its estimate. The higher the SDR the better X_true and X agree"""
    return 10*np.log10(np.sum(X_true**2)**0.5/np.sum((X_true-X)**2)**0.5)

def chi(image, model):
    return image.shape[0]/image.size*(np.sum((image - model)**2, axis = (-2,-1))/scarlet.wavelet.mad_wavelet(image)**2)

class simulation:
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
            hr, lr, wcs_hr, wcs_lr, psf_hr, psf_lr, shifts, ks, gs_lr, gs_hr, seds_hr, seds_lr = mk_scene(self.hr_dict, 
                                                                      self.lr_dict, 
                                                                      self.cat, 
                                                                      (self.n_hr,self.n_hr), 
                                                                      (self.n_lr,self.n_lr), 
                                                                      ns, gal_type = 'real')
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
            
            
def get_flux(mag, exp, zero_point):
    """Computes the flux for an object at a given magnitude.
    
    Parrameters
    -----------
    mag: float
        magnitude of the object
    """
    
    return exp*zero_point*10**(-0.4*(mag-24))
    
    
    
    
    
    
    
    
    
            
                