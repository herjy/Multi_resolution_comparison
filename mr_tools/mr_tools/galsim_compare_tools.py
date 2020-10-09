import galsim
import scarlet
from . import pictures

center_ra = 19.3*galsim.degrees     # The RA, Dec of the center of the image on the sky
center_dec = -33.1*galsim.degrees


def mk_scene(hr_dict,
             lr_dict,
             cat,
             shape_hr,
             shape_lr,
             n_gal,
             gal_type,
             random_seds = True,
             noise = True,
             pt_fraction = 0,
             magmin=20,
             magmax=29,
             index = None,
             use_cat = True,
             shift=True):
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
    pt_fraction: float
            fraction of point sources
    """
    pic_hr = pictures.Pictures(hr_dict, shape_hr, cat=cat, pt_fraction = pt_fraction)
    pic_lr = pictures.Pictures(lr_dict, shape_lr, cat=cat)

    pic_hr.make_cube(n_gal,
                     gal_type=gal_type,
                     random_seds=random_seds,
                     noisy=noise,
                     magmin=magmin,
                     magmax=magmax,
                     index=index,
                     use_cat=use_cat,
                     shifty=shift)
    pic_lr.make_cube(n_gal,
                     picture=pic_hr,
                     gal_type=gal_type,
                     random_seds=random_seds,
                     noisy=noise,
                     magmin=magmin,
                     magmax=magmax,
                     index=index,
                     use_cat=use_cat,
                     shifty=shift)

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
    obs_hr = scarlet.Observation(im_hr, [channels[1]], wcs=wcs_hr, psf=psf_hr)
    obs_lr = scarlet.Observation(im_lr, [channels[0]], wcs=wcs_lr, psf=psf_lr)

    # Keep the order of the observations consistent with the `channels` parameter
    # This implementation is a bit of a hack and will be refined in the future
    obs = [obs_lr, obs_hr]

    frame = scarlet.Frame.from_observations(obs, obs_id = 1, coverage = coverage)
    return obs, frame

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
    interp_gal = conv_gal.drawImage(nx=n_lr,ny=n_lr, scale=h_lr, method='no_pixel',)

    return interp_gal





