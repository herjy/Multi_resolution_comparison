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
    def __init__(self, survey, shape, cat = cat, npsf = 41):
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
        self.psfs_obj, self.psfs = self.make_psf(npsf)

        if self.survey['sky'] is not None:
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

        #Creating WCS
        w = WCS.WCS(naxis=2)
        galsim_wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w.wcs.crpix = galsim_wcs.crpix
        w.wcs.pc = galsim_wcs.cd
        w.wcs.crval = [galsim_wcs.center._ra._rad, galsim_wcs.center._dec._rad]
        w.array_shape = self.shape

        return w

    def make_galaxy(self, k = None, shift = 0, angle = 0, point = False, gal = None, gal_type = 'real', smooth = True):
        """Creates the galsim object
        k: int
            index of the galaxy to draw from the catalog
        shift: tuple
            2D position of the galaxy in the stamp
        angle: float
            Rotation angle of the image
        point: Boolean
            If true, draws a point source instead of a galaxy
        gal: galsim.Galaxy object
            provides the galaxy object beforehand
        gal_type: string
            eal or parametric profile
        """

        if gal == None:
            #Rotation angle
            angle = galsim.Angle(angle,galsim.radians)

            if point:
                gal = galsim.Gaussian(sigma = 0.02).withFlux(1)
            if not point:
                #Galaxy profile
                gal = cat.makeGalaxy(k, gal_type = gal_type, noise_pad_size= 0).withFlux(1)
                if smooth == True:
                    gal = galsim.Convolve(gal, galsim.Gaussian(sigma=self.pix))

            gal = gal.shift(dx=shift[0], dy=shift[1])


        # np array of the image before psf convolution
        unconvolved = gal.drawImage(nx=self.shape[0],ny=self.shape[1],
                     use_true_center = True, method = 'no_pixel',
                     scale = self.pix, dtype = np.float64).array

        galaxy = []
        for i in range(len(self.channels)):
            #The actual image with psf
            galaxy.append(np.array(galsim.Convolve(gal, self.psfs_obj[i]).drawImage(nx=self.shape[0],
                                                                                 ny=self.shape[1],
                                                                                 use_true_center = True,
                                                                                 method = 'no_pixel',
                                                                                 scale = self.pix,
                                                                                 dtype = np.float64).array))
        return np.array(galaxy), unconvolved, gal

    def make_cube(self, ns, picture = None, gal_type = 'real', pt_fraction = 0.1):
        """ Creates a cube of images of galaxies randomly spread across a postage stamp

        Parameters
        ----------
        ns: int
            Number of sources per patch
        picture: Picture object
            if provided, the attributes of picture are used to draw the cube
        gal_type: string
            The type of galaxies to draw ('real' or 'parametric')
        pt_fraction: float

        :return:
        """
        if picture == None:
            self.ks, self.shifts, self.gals = [], [], []
        else:
            self.ks = picture.ks
            self.gals = picture.gals
            self.shifts = (np.array(picture.shifts)-self.pix)/(np.array(self.shape)/2)

        self.galaxies, self.seds, self.mags = [], [], []
        cube = np.zeros((len(self.channels), *self.shape))
        for i in range(ns):

            # Magnitudes
            mag = np.random.rand(len(self.channels)) * 10 + 18
            # SED
            sed = self.get_flux(mag)

            if picture == None:
                #Morphology
                k = np.int(np.random.rand(1)*len(self.cat))
                #Position
                shift = ((np.random.rand(2)-0.5) * self.shape * self.pix * 2 / 3)
                #Point source?
                toss = (np.random.rand(1) < pt_fraction)
                if toss:
                    k = 'point'

                galaxy, unconvolved, gal = self.make_galaxy(k = k,
                                                       shift = shift,
                                                       point = toss,
                                                       gal_type = gal_type)

                self.galaxies.append(unconvolved)
                self.seds.append(sed)
                self.ks.append(k)
                self.mags.append(mag)
                self.gals.append(gal)
                self.shifts.append([shift[0] / self.pix + self.shape[0] / 2,
                                    shift[1] / self.pix + self.shape[1] / 2])
            else:

                galaxy, unconvolved, gal = self.make_galaxy(k=None,
                                                            shift=None,
                                                            point=None,
                                                            gal=self.gals[i],
                                                            gal_type=gal_type)

                self.galaxies.append(unconvolved)
                self.seds.append(sed)
                self.mags.append(mag)

            cube += sed[:, None, None] * galaxy



        self.cube = cube + np.random.randn(*cube.shape)*np.sqrt(self.noise[:, None, None])



    @property
    def images(self):
        return self._images

    def get_flux(self, mag):
        """Computes the flux for an object at a given magnitude.

        Parrameters
        -----------
        mag: float
            magnitude of the object
        """

        return self.survey['exp_time']*self.survey['zero_point']*10**(-0.4*(mag-24))

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
            psfs.append(psfs_obj[-1].drawImage(nx=npsf,
                                          ny=npsf,
                                          method = 'real_space',
                                          use_true_center = True,
                                          scale = self.pix).array)
        return psfs_obj, psfs


