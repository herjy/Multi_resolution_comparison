import galsim
from astropy import wcs as WCS
import numpy as np
import scipy.signal as scp
import scarlet
from scarlet_extensions.initialization.detection import Data
import pickle
import matplotlib.pyplot as plt
from . import pictures

center_ra = 19.3*galsim.degrees     # The RA, Dec of the center of the image on the sky
center_dec = -33.1*galsim.degrees


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

    pic_hr = pictures.Pictures(hr_dict, shape_hr, cat = cat)
    pic_lr = pictures.Pictures(lr_dict, shape_lr, cat = cat)

    pic_hr.make_cube(n_gal, gal_type = gal_type)
    pic_lr.make_cube(n_gal, pic_hr, gal_type = gal_type)

    return pic_hr, pic_lr

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











