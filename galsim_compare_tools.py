import galsim
from astropy import wcs as WCS
import numpy as np
import scarlet

center_ra = 19.3*galsim.hours     # The RA, Dec of the center of the image on the sky
center_dec = -33.1*galsim.degrees
def load_surveys():
    """Creates dictionaries for the HST, EUCLID, WFRIST, HCS anf LSST surveys
    that contain their names, pixel sizes and psf fwhm in arcseconds"""
    pix_wfirst = 0.11
    pix_RUBIN = 0.2
    pix_HST = 0.06
    pix_Euclid = 0.1
    pix_HSC = 0.167
    
    #Sigma of the psf profile in arcseconds.
    sigma_wfirst = 1.69*0.11 #https://arxiv.org/pdf/1702.01747.pdf Z-band
    sigma_RUBIN = 0.7 #https://www.lsst.org/about/camera/features
    sigma_Euclid = 0.16 #https://sci.esa.int/documents/33859/36320/1567253682555-Euclid_presentation_Paris_1Dec2009.pdf
    sigma_HST = 0.074 #Source https://hst-docs.stsci.edu/display/WFC3IHB/6.6+UVIS+Optical+Performance#id-6.6UVISOpticalPerformance-6.6.1 800nm
    sigma_HSC = 0.62 #https://hsc-release.mtk.nao.ac.jp/doc/ deep+udeep
    
    EUCLID = {'name': 'EUCLID', 'pixel': pix_Euclid ,'psf': sigma_Euclid}
    HST = {'name': 'HST', 'pixel': pix_HST,'psf': sigma_HST}
    HSC = {'name': 'HSC', 'pixel': pix_HSC,'psf': sigma_HSC}
    WFIRST = {'name': 'WFIRST', 'pixel': pix_wfirst,'psf': sigma_wfirst}
    RUBIN = {'name': 'RUBIN', 'pixel': pix_RUBIN,'psf': sigma_RUBIN}
    
    return HST, EUCLID, WFIRST, HSC, RUBIN

HST, EUCLID, WFIRST, HSC, RUBIN = load_surveys()

def mk_wcs(theta, pix, center, shape):
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
    galfit_wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.crpix = galfit_wcs.crpix

    w.wcs.pc = galfit_wcs.cd
    w.wcs.crval = [galfit_wcs.center._ra._rad, galfit_wcs.center._dec._rad]
    w.array_shape = shape
    return w
    

def mk_sim(k, hr_dir, lr_dir, shape_hr, shape_lr, npsf, cat):
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
    
    #Galaxy profile
    gal = cat.makeGalaxy(k, gal_type = 'real', noise_pad_size=shape_lr[0] * pix_lr*0)
    
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
    im_hr = galsim.Convolve(gal, psf_hr_int).drawImage(nx=shape_hr[0],ny=shape_hr[1], 
                                                       use_true_center = True, method = 'no_pixel',
                                                   scale = pix_hr, dtype = np.float64)
    # Convolve galaxy profile by PSF, rotate and sample at low resolution
    im_lr = galsim.Convolve(gal.rotate(angle), psf_lr_int).drawImage(nx=shape_lr[0],ny=shape_lr[1], 
                                                                     use_true_center = True, method = 'no_pixel',
                                                                 scale = pix_lr, dtype = np.float64)
    
    # Make WCSs
    im_hr.wcs = mk_wcs(0, pix_hr, galsim.PositionD(im_hr.true_center), shape_hr)
    im_lr.wcs = mk_wcs(0, pix_lr, galsim.PositionD(im_lr.true_center), shape_lr)
   
    return im_hr, im_lr, psf_hr[None,:,:], psf_lr[None,:,:], theta

def setup_scarlet(data_hr, data_lr, psf_hr, psf_lr, channels, coverage = 'union'):
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
    im_hr = data_hr.array[None, :, :]
    im_lr = data_lr.array[None, :, :]
    
    # define two observation objects and match to frame  
    obs_hr = scarlet.Observation(im_hr, wcs=data_hr.wcs, psfs=psf_hr, channels=[channels[1]])
    obs_lr = scarlet.Observation(im_lr, wcs=data_lr.wcs, psfs=psf_lr, channels=[channels[0]])

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
    im_hr = data_hr.array[None, :, :]
    im_lr = data_lr.array[None, :, :]
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
