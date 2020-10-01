import galsim
from astropy import wcs as WCS
import numpy as np
import scipy.signal as scp
import scarlet
from scarlet_extensions.initialization.detection import Data
import pickle
import matplotlib.pyplot as plt

data_dir='/Users/remy/Desktop/LSST_Project/GalSim/examples/data'

cat = galsim.COSMOSCatalog(dir=data_dir, file_name = 'real_galaxy_catalog_23.5_example.fits')


class Pictures:
    """ Class that draws simulated images of the sky at random given a dictionary 
    """
    def __init__(self, survey, shape, cat = cat) 
        """
        survey: dict
        Properties of the imaging survey for which we want to generate images
        shape: tuple
            2D shape of the output stamp image
        cat: Catalog
            Galsim catalog of galaxies
    
        """
        
        self.cat = cat
        self.survey = survey
        self.pix = self.survey['pixel']
        self.channels = self.survey['channels']
        self.shape = shape
        image = galsim.Image(self.shape[0], self.shape[1], scale=self.pix)
        
        self.wcs = self.mk_wcs(galsim.PositionD(image.true_center))
        self.psfs_obj, self.psfs = make_psf(41)
        self.noise = self.get_flux(self.survey['sky'])*self.pix**2
        
    def mk_wcs(self, 
               center, 
               theta = 0, 
               sky_center = galsim.CelestialCoord(ra=19.3*galsim.degrees, 
                                                  dec=-33.1*galsim.degrees)):
        '''Creates wcs for an image
        
        Parameters
        ----------
        theta: float
            rotation angle for the image
        center: galsim.PositionD
            position of the reference pixel used as the center of the affin transform for the wcs
        
        Returns
        -------
        wcs: WCS
        '''
        #Affine transformation
        dudx = np.cos(theta) * self.pix
        if theta == 0:
            dudy = 0
            dvdx = 0
        else:
            dudy = -np.sin(theta) * self.pix
            dvdx = np.sin(theta) * self.pix
        dvdy = np.cos(theta) * self.pix
        
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=center)
        #Image center
        sky_center = 
        
        #Creating WCS
        w = WCS.WCS(naxis=2)
        galsim_wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w.wcs.crpix = galsim_wcs.crpix
        w.wcs.pc = galsim_wcs.cd
        w.wcs.crval = [galsim_wcs.center._ra._rad, galsim_wcs.center._dec._rad]
        w.array_shape = self.shape
        
        return w
    
    def make_galaxy(self, k = None, shift = 0, angle = 0, point = False, gal = None):
        """Creates the galsim object
        k: int
            index of the galaxy to draw from the catalog
        shift: tuple
            2D position of the galaxy in the stamp
        angle: float
            Rotation angle of the image
        point: Boolean
            If true, draws a point source instead of a galaxy
        """
        
        if gal == None:
            #Rotation angle
            angle = galsim.Angle(angle,galsim.radians)
        
            if point:
                gal = galsim.Gaussian(sigma = 0.02).withFlux(1)
            if not point:
                #Galaxy profile
                gal = cat.makeGalaxy(k, gal_type = gal_type, noise_pad_size=shape_lr[0] * pix_lr * 0).withFlux(1)
                if smooth == True:
                    gal = galsim.Convolve(gal, galsim.Gaussian(sigma=pix_hr))
        
            gal = gal.shift(dx=shift[0], dy=shift[1])
        
        
        # np array of the image before psf convolution
        unconvolved = gal.drawImage(nx=self.shape[0],ny=self.shape[1], 
                     use_true_center = True, method = 'no_pixel',
                     scale = self.pix, dtype = np.float64).array
        
        galaxy = []
        for i in range(len(self.channels));
            #The actual image with psf
            galaxy.append(np.array(galsim.Convolve(gal, self.psfs_obj[i]).drawImage(nx=self.shape[0],
                                                                                 ny=self.shape[1], 
                                                                                 use_true_center = True, 
                                                                                 method = 'no_pixel',
                                                                                 scale = self.pix, 
                                                                                 dtype = np.float64).array))
        return np.array(galaxy), unconvolved, gal
        
    def make_cube(self, ns, gals = None):
        
        self.galaxies, self.seds, self.ks, self.mags, self.gals, self.shifts = [], [], [], [], [], []
        cube = np.zeros((len(self.channels, *self.shape)))
        for i in range(ns):
            #Morphology
            k = np.int(np.random.rand(1)*len(self.cat))
            #Position
            shift = ((np.random.rand(2)-0.5) * self.shape * self.pix * 2 / 3)
            #Magnitudes
            mag = np.random.rand(len(self.channels))*10 + 18
            #SED
            sed = self.get_flux(mags)
            
            toss = (np.random.rand(1) < 0.1)
            if toss:
                k = 'point'
                sed_lr *= 100
                
            galaxy, unconvolved = self.make_galaxy(k = k, shift = shift, point = toss, mag = mag, gal = gals[i])
            
            cube += sed[:, None, None] * galaxy
            
            self.galaxies.append(unconvolved)
            self.seds.append(sed)
            self.ks.append(k)
            self.mags.append(mag)
            self.gals.append(gal)
            self.shifts.append(shift)
        
        self._cube = cube + np.random.randn(cube.shape)*self.noise
            
            
            
    @property
    def self.images(self):
        return self._images
    
    def get_flux(self, mag):
    """Computes the flux for an object at a given magnitude.
    
    Parrameters
    -----------
    mag: float
        magnitude of the object
    """
    
    self.flux = self.survey['exp_time']*self.survey['zero_point']*10**(-0.4*(mag-24))
    
    def make_psf(self, npsf = 41):
        
        psfs_obj = []
        psfs = []
        for i in range(len(self.channels)):
            psf_init = galsim.Gaussian(sigma = self.survey['psf'][i]).withFlux(1.)
            ## Draw PSF
            psf = psf_init.drawImage(nx=npsf,
                                ny=npsf, 
                                method = 'real_space',
                                use_true_center = True, 
                                scale = self.pix).array
            ## Make sure PSF vanishes on the edges of a patch that has the shape of the initial npsf
            psf = psf-psf[0, int(npsf/2)]*2
            psf[psf<0] = 0
            psf /= np.sum(psf)
            ## Interpolate the new 0-ed psf 
            psfs_obj.append(galsim.InterpolatedImage(galsim.Image(psf), scale = self.pix).withFlux(1.))
            ## Re-draw it (with the correct fulx)
            psfs.append(psf_obj.drawImage(nx=npsf,
                                          ny=npsf, 
                                          method = 'real_space',
                                          use_true_center = True, 
                                          scale = self.pix).array)
            return psf_oj, psf
            
            
        
    
    
def mk_sim(k, hr_dir, lr_dir, shape_hr, shape_lr, npsf, cat, shift = (0,0), gal_type = 'real', smooth = False, point = False):

   
    return im_hr.array, im_lr.array, wcs_hr, wcs_lr, psf_hr[None,:,:], psf_lr[None,:,:], theta, i_hr.array, i_lr.array
    