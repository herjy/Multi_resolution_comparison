import galsim
from astropy import wcs as WCS
import numpy as np
import astropy
import os

data_dir='/Users/remy/Desktop/LSST_Project/GalSim/examples/data'

cat0 = galsim.COSMOSCatalog(dir=data_dir, file_name = 'real_galaxy_catalog_23.5_example.fits')
mag_cat_name = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sample_input_catalog.fits')
mag_cat = astropy.table.Table.read(mag_cat_name, format = 'fits')
mags = np.array([mag_cat["u_ab"],
        mag_cat["g_ab"],
        mag_cat["r_ab"],
        mag_cat["i_ab"],
        mag_cat["z_ab"],
        mag_cat["y_ab"]]).T

l_cat = len(mag_cat)

class Pictures:
    """ Class that draws simulated images of the sky at random given a dictionary
    """
    def __init__(self, survey, shape, cat = cat0, npsf = 41, pt_fraction = 0.1):
        """
        survey: dict
        Properties of the imaging survey for which we want to generate images
        shape: tuple
            2D shape of the output stamp image
        cat: Catalog
            Galsim catalog of galaxies
        npsf: int
            size of the psf's postage stamp
        pt_fraction: float
            fraction of point sources

        """
        self.point = galsim.Gaussian(flux = 1, sigma = 1.e-8)
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
        self.pt_fraction = pt_fraction

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
        sky_center: galsim.CelestialCoord
            Reference coordinates of the center of the image in celestial coordinates
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

    def make_galaxy(self,
                    k=None,
                    shift=[0,0],
                    point=False,
                    gal=None,
                    gal_type='real',
                    smooth=True):
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

            if point:
                gal = self.point

            if not point:
                #Galaxy profile
                gal = self.cat.makeGalaxy(k, gal_type = gal_type, noise_pad_size= 0).withFlux(1)

                if smooth == True:
                    gal = galsim.Convolve(gal, galsim.Gaussian(sigma=2*self.pix))

                if smooth == True:
                    gal = galsim.Convolve(gal, galsim.Gaussian(sigma=1*self.pix))
            gal = gal.shift(dx=shift[0], dy=shift[1])


        param = self.cat.makeGalaxy(k, gal_type='parametric', noise_pad_size=0).withFlux(1)
        if smooth == True:
            param = galsim.Convolve(param, galsim.Gaussian(sigma=2 * self.pix))
        param = param.shift(dx=shift[0], dy=shift[1])
        parametric = param.drawImage(nx=self.shape[0],
                                     ny=self.shape[1],
                                     use_true_center=True,
                                     method='real_space',
                                     scale=self.pix,
                                     dtype=np.float64).array.T

        # np array of the image before psf convolution
        unconvolved = gal.drawImage(nx=self.shape[0],
                                    ny=self.shape[1],
                                    use_true_center = True,
                                    method='real_space',
                                    scale=self.pix,
                                    dtype=np.float64).array


        galaxy = []
        for i in range(len(self.channels)):
            #The actual image with psf
            galaxy.append(galsim.Convolve(gal, self.psfs_obj[i]).drawImage(nx=self.shape[0],
                                  ny=self.shape[1],
                                  use_true_center = True,
                                  method='fft',
                                  scale=self.pix,
                                  dtype=np.float64).array)

        return np.array(galaxy), unconvolved, gal, parametric

    def make_cube(self,
                  ns,
                  picture=None,
                  gal_type='real',
                  random_seds=True,
                  noisy=True,
                  magmax=30,
                  magmin=20,
                  index=None,
                  use_cat=True,
                  shifty=True):
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

        self.galaxies, self.seds, self.mags, self.parametrics = [], [], [], []

        cube = np.zeros((len(self.channels), self.shape[0], self.shape[1]))
        for i in range(ns):
            # Magnitudes
            if picture != None:
                if use_cat:
                    magmini = 19.9
                    while magmini <magmin:
                        ind = np.int(np.random.rand(1)*l_cat)
                        mag0 = np.random.rand(1) * (magmax-magmin) + magmin

                        mag = mags[ind] + mag0 - np.min(mags[ind])
                        magmini = np.min(mag)
                else:
                    mag0 = np.random.rand(1) * (magmax-magmin) + magmin
                    mag = mag0 + np.random.randn(len(self.channels))
            else:
                mag = np.random.rand(len(self.channels)) * (magmax-magmin) + magmin

            if index == None:
                # Morphology
                k = 62
                while k == 62:
                    k = np.int(np.random.rand(1) * len(self.cat))

            else:
                k = index
            # Position
            if shifty:
                shift = ((np.random.rand(2) - 0.5) * self.shape * self.pix * 2 / 3)
            else:
                shift = [0,0]

            # Point source?
            toss = (np.random.rand(1) < self.pt_fraction)
            # SED
            if random_seds:
                sed = self.get_flux(mag)
            else:
                sed = np.ones(mag.shape)
            if picture == None:
                if toss:
                    k = 'point'
                galaxy, unconvolved, gal, parametric = self.make_galaxy(k=k,
                                                       shift=shift,
                                                       point=toss,
                                                       gal_type=gal_type)

                self.galaxies.append(unconvolved)
                self.parametrics.append(parametric)
                self.seds.append(sed)
                self.ks.append(k)
                self.mags.append(mag)
                self.gals.append(gal)
                self.shifts.append([shift[0] / self.pix + self.shape[0] / 2 - 0.5,
                                    shift[1] / self.pix + self.shape[1] / 2 - 0.5])
            else:

                galaxy, unconvolved, gal, parametric = self.make_galaxy(k=k,
                                                            shift=shift,
                                                            point=None,
                                                            gal=self.gals[i],
                                                            gal_type=gal_type)

                self.galaxies.append(unconvolved)
                self.parametrics.append(parametric)
                self.seds.append(sed)
                self.mags.append(mag)

            cube += sed[:, None, None] * galaxy

        cube += np.random.randn(*cube.shape) * np.sqrt(self.noise[:, None, None])*noisy
        self.cube = cube/np.max(cube)*100

    def get_flux(self, mag):
        """Computes the flux for an object at a given magnitude.

        Parrameters
        -----------
        mag: float
            magnitude of the object
        """

        return self.survey['exp_time']*self.survey['zero_point']*10**(-0.4*(mag-24))


    def make_psf(self, npsf=41):

        psfs_obj = []
        psfs = []
        for i in range(len(self.channels)):
            psf_init = galsim.Gaussian(sigma = self.survey['psf'][i]).withFlux(1.)

            ## Draw PSF
            psf = psf_init.drawImage(nx=npsf,
                                ny=npsf,
                                method='real_space',
                                use_true_center=True,
                                scale=self.pix).array

            ## Interpolate the new 0-ed psf
            psfs_obj.append(psf_init)

            ## Re-draw it (with the correct fulx)
            psfs.append(psf)
        return psfs_obj, psfs


