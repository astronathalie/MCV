from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus, aperture_photometry, EllipticalAperture, EllipticalAnnulus
from photutils.centroids import centroid_sources, centroid_2dg
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
import pandas as pd
import os
import shlex
import glob
from astropy.time import Time
from photutils.profiles import RadialProfile
import subprocess
import warnings
from astroquery.vizier import Vizier
warnings.filterwarnings('ignore')
from astropy.visualization import ZScaleInterval, ImageNormalize

from pathlib import Path
from datetime import date

from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import coordinates as coord
import configparser
import array
from astroquery.jplhorizons import Horizons
from astroquery.simbad import Simbad
from datetime import datetime

###################################
### Photometry Performing Class ###
###################################

class PhotometryPipeline:
    """
    Encapsulates relative photometry steps: performing photometry,
    calculating aperture-scale ratios, and finding magnitudes.
    """
    def __init__(self,
                 file_dir: str,
                 star_list_pix: list,
                 ref_image: str = None,
                 target_name: str = None,
                 aperture_radius: float = 8,
                 annulus_inner: float = 22,
                 annulus_outer: float = 30,
                 afactor: float = 1.0,
                 bfactor: float = 1.0,
                 gap: float = 5.0,
                 theta: float = 0,
                 gain: float = None,
                 rdnoise: float = 2.8):
        self.file_dir = file_dir
        self.star_list_pix = star_list_pix
        self.ref_image = ref_image
        self.target_name = target_name
        self.ap_radius = aperture_radius
        self.an_inner = annulus_inner
        self.an_outer = annulus_outer
        self.a = afactor
        self.b = bfactor
        self.gap = gap
        self.theta = theta
        self.gain = gain
        self.rdnoise = rdnoise
        '''The above parameters are default values and can be changed if needed
        The value are set when you run it.
        The gain and rdnoise are set to correct values (MAY 2025 RLMT) but can be changed if needed
        GAIN will read from the header if set to None, this is currently set to use 4*egain due to binning
        '''

    def _load_images(self, filt):
        path = self.file_dir + '/'
        patterns = ['*.fts', '*.fts.fz', '*.fit', '*.fits']
        files = []
        for p in patterns:
            files += glob.glob(os.path.join(path, '*_' + filt + '_') + p)
        return sorted(files)

    def _read_header(self, fname):
        try:
            hdr = fits.getheader(fname, ext=1)
        except Exception:
            hdr = fits.getheader(fname, ext=0)
        if self.gain is None:
            egain = hdr.get('EGAIN', 1)
            self.gain = 4 * egain
        return hdr

    def perform_phot(self, filt,
                     save=True,
                     plot=True,
                     save_ref_star_coords=False,
                     display_apertures=False,
                     HJD = False,
                     use_hdr_ap = False,
                     large_ap = False,
                     errorshow = True,
                     title = None,
                     non_sidereal = False):
        '''
        Perform relative photometry on a set of images
        Variable FWHM photometry is implemented in different definitions.
        By default the code will save to a csv file and plot results, this can be turned off with flags
        '''
        images = self._load_images(filt)
        if not images:
            raise FileNotFoundError(f"No images for filter {filt}")
        ref = self.ref_image or images[0]
        hdr = self._read_header(ref)
        wcs = WCS(hdr)
        # convert pix->world->pix for all
        coords = [wcs.all_pix2world(x, y, 0) for x, y in self.star_list_pix]
        # add the target star to the list
        if self.target_name is None:
            try:
                target_name = hdr['BLKNAME']
            except KeyError:
                target_name = hdr['OBJECT']
            print('Target Name:', target_name)
        else:
            target_name = self.target_name
            print('Target Name:', target_name)
        # Convert the target coordinates to degrees
        if not non_sidereal:
            Simbad.add_votable_fields('pmra', 'pmdec')
            result = Simbad.query_object(target_name)
            target = SkyCoord(
                ra=result['ra'][0],
                dec=result['dec'][0],
                unit=(u.deg, u.deg),
                pm_ra_cosdec=result['pmra'][0]*u.mas/u.yr,
                pm_dec=result['pmdec'][0]*u.mas/u.yr,
                obstime=Time('J2000.0'),
                frame='icrs'
            )
            target = target.apply_space_motion(new_obstime=Time(datetime.now().isoformat()))
            ra = target.ra.deg
            dec = target.dec.deg
            print('Target RA:', ra)
            print('Target Dec:', dec)
            coords = [(ra, dec)] + coords
        all_results = []

        # Loop over each image file
        for im in images:
            hdr = self._read_header(im)
            exptime = hdr['EXPTIME']
            jd = hdr['JD'] + exptime/(2*86400)
            if non_sidereal:
                non_sid_obj=Horizons(id=non_sid_id, location=non_sid_loc, epochs=jd)
                eph=non_sid_obj.ephemerides()
                ra=eph['RA'][0]
                dec=eph['DEC'][0]
                print('Target RA:', ra)
                print('Target Dec:', dec)
                coords = [(ra, dec)] + [wcs.all_pix2world(x, y, 0) for x, y in self.star_list_pix]
            data = fits.getdata(im)
            wcs = WCS(hdr)
            xy = [wcs.all_world2pix(ra, dec, 0) for ra, dec in coords]
            if save_ref_star_coords:
                star_list_saving = pd.DataFrame(coords, columns=['RA', 'Dec'])
                target_name = target_name.replace(' ', '_')
                star_list_saving.to_csv(self.file_dir + '/' + title + '_ref_star_coords.csv', index=False)
                print('Saving reference star coordinates to', title + '_ref_star_coords.csv')
            exptime = hdr['EXPTIME']
            if HJD:
                # Get the Julian Date from the header and add the half exposure time to get the mid-exposure time
                exptime = hdr['EXPTIME']
                exptimedays = exptime/(24*3600)
                addjd = exptimedays/2
                jd = hdr['HJD'] + addjd
            else:
                jd = hdr['JD'] + exptime/(2*86400)
            file_results = {'file': im, 'Julian_Date': jd}
            sigclip = SigmaClip(sigma=3., maxiters=10)
            skip_file = False
            # Begin WCS Check:
            try:
                header=self._read_header(im)
                wcs = WCS(header)
                # If WCS has no celestial axes, this will fail
                if not wcs.has_celestial:
                    raise ValueError("No celestial WCS")
                # Try a world → pixel transform as a functional test
                x1, y1 = wcs.all_world2pix(ra,dec,0)
                # Catch NaNs
                if not (x1 == x1 and y1 == y1):
                    raise ValueError("NaN pixel coordinates")
            except Exception:
                print("WCS Failure; skipping file: ", im)
                skip_file=True
                continue
            for i, (x, y) in enumerate(xy):
                if use_hdr_ap:
                    ap_r = hdr['AP_R']
                    an_i = hdr['AN_R1']
                    an_o = hdr['AN_R2']
                else:
                    ap_r = self.ap_radius
                    an_i = self.an_inner
                    an_o = self.an_outer
                if large_ap:
                    ap_r = ap_r * 2.5
                ap = CircularAperture((x, y), r=ap_r)
                an = CircularAnnulus((x, y), r_in=an_i, r_out=an_o)
                # Begin Saturation Check
                mask = ap.to_mask(method='center')
                cutout = mask.cutout(data)
                ap_pixels = cutout[mask.data.astype(bool)]
                # Check for saturation
                if np.any(ap_pixels > 60000):
                    print(f"Saturated pixel found in {im}!")
                    skip_file = True
                bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                #ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                ap_stats = ApertureStats(data, ap)
                #print(x,y)
                # Recentroid the aperture
                if bkg_stats.median > ap_stats.mean:
                    print('Background is too high, recentering')
                    an = CircularAnnulus((x, y), r_in= an_i + 10, r_out=an_o + 10)
                    bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                sub = data - bkg_stats.median
                x0, y0 = x, y
                try:
                    x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                    if np.isnan(x) or np.isnan(y):
                        raise ValueError
                    x, y = float(x), float(y)
                except Exception:
                    print(f"  → centroid failed for star {i} on image {im}, trying again")
                    x, y = x0, y0
                    ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                    x, y = ap_stats.centroid
                # try:
                #     x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                #     if np.isnan(x) or np.isnan(y):
                #         raise ValueError
                #     x, y = float(x), float(y)
                # except Exception:
                #     x, y = x0, y0


                aperture = CircularAperture((x, y), r=ap_r)
                annulus_aperture = CircularAnnulus((x, y), r_in=an_i, r_out=an_o)

                # Perform aperture photometry
                phot_table = aperture_photometry(data, aperture)
                bkgstats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)

                # Calculate the background in the annulus
                bkg_mean = bkgstats.mean
                bkg_sum = bkg_mean * aperture.area

                # Subtract the pedestal from the background for the error calculation
                pedestal = hdr.get('PEDESTAL', 0)  # Default pedestal value if not found
                bkg_mean_nopedestal = bkg_mean-pedestal
                bkg_sum_nopedestal = bkg_mean_nopedestal * aperture.area

                # Subtract the background from the aperture photometry
                source_sum = phot_table['aperture_sum'][0]*self.gain - bkg_sum*self.gain

                # Check if the  is negative and skip if it is
                if source_sum < 0:
                    print(f"  star {i}, pos=({x:.1f},{y:.1f}), "
                          f"ap_sum={phot_table['aperture_sum'][0]:.1f}, "
                          f"bkg_mean={bkg_mean:.1f}, bkg_sum={bkg_sum:.1f}, "
                          f"gain={self.gain}")
                    skip_file = True
                    break
                # if source_sum < 0:
                #     print(f"Skipping {im} due to negative , image should be inspected.")
                #     skip_file = True
                #     break

                # if source_sum < 0:
                #     source_sum = 0.000001
                #     print(f" {im} for is negative, setting to 0.000001")

                # Error calculation (Poisson noise + background noise + read noise)

                error = np.sqrt((phot_table['aperture_sum'][0]-pedestal*aperture.area)*self.gain + ((aperture.area)/annulus_aperture.area)*bkg_sum_nopedestal*self.gain + aperture.area*self.rdnoise**2 + aperture.area**2/annulus_aperture.area*self.rdnoise**2)

                # Optionally turn into a magnitude (not used here but useful for reference)
                source_mag = -2.5 * np.log10(source_sum / exptime)
                source_mag_err = 1.0857 * error / source_sum

                # Store the results with dynamic column names
                file_results[f'star_{i}_x'] = x
                file_results[f'star_{i}_y'] = y
                file_results[f'star_{i}_flux'] = source_sum
                file_results[f'star_{i}_error'] = error
                file_results[f'star_{i}_background'] = bkg_sum

            # Optionally, display the image with the apertures and annuli (set flag to True)
            # This is useful for checking the positions of the stars are correct
            if display_apertures:
                plt.imshow(data, vmin=np.percentile(data, 5), vmax=np.percentile(data, 99), cmap='viridis')
                for (x, y) in self.star_list_pix:
                    aperture = CircularAperture((x, y), r=ap_r)
                    annulus_aperture = CircularAnnulus((x, y), r_in=an_i, r_out=an_o)
                    aperture.plot(color='blue', lw=1.5)
                    annulus_aperture.plot(color='red', lw=1.5)
                plt.show()

            # Append the results for this file to the list of all results
            if not skip_file:
                all_results.append(file_results)
            if skip_file:
                print(f"Skipping {im} due to negative !, image should be inspected.")
                continue

        # Convert the results to a DataFrame for easy analysis
        results_df = pd.DataFrame(all_results)

        # Rename the columns to remove the 'star_0_' prefix for the first star
        # and replace it with 'target_' for clarity
        results_df.rename(
        columns=lambda c: c.replace("star_0_", "target_")
        if c.startswith("star_0_") else c,
        inplace=True
        )

        # add all the comparison stars together
        results_df['total_flux'] = results_df['star_1_flux'] + results_df['star_2_flux'] + results_df['star_3_flux'] + results_df['star_4_flux'] + results_df['star_5_flux']

        # calculate the relative flux of the target star
        results_df['target_rel_flux']=results_df['target_flux']/results_df['total_flux']

        # calculate the relative flux of the comparison stars
        for i in range(1,6):
            results_df[f'star_{i}_relflux'] = results_df[f'star_{i}_flux']/results_df['total_flux']

        # calculate the error on the total flux
        total_flux_err = np.sqrt(results_df['star_1_error']**2 + results_df['star_2_error']**2 + results_df['star_3_error']**2 + results_df['star_4_error']**2 + results_df['star_5_error']**2)

        #calculate the error on the relative flux
        results_df['target_relerror'] = (results_df['target_flux']/results_df['total_flux'])*(np.sqrt((results_df['target_error']/results_df['target_flux'])**2 + (total_flux_err/results_df['total_flux'])**2))

        # Normalize the relative flux
        mean_rel_flux, _, _, = sigma_clipped_stats(results_df['target_rel_flux'], sigma=  2.0)
        results_df['norm_target_rel_flux'] = results_df['target_rel_flux']/mean_rel_flux
        results_df['norm_target_rel_flux_error'] = results_df['target_relerror']/mean_rel_flux

        # Save the results to a CSV file
        if save:
            results_df.to_csv(self.file_dir + '/' + title + '_Results_' + filt + '.csv', index=False)

        if plot:
            #Plot the relative flux
            #plt.plot(results_df['Julian_Date'], results_df['target_rel_flux'], 'o')
            if errorshow:
                plt.errorbar(results_df['Julian_Date'], results_df['target_rel_flux'], yerr=results_df['target_relerror'], fmt='o')
            else:
                plt.scatter(results_df['Julian_Date'], results_df['target_rel_flux'])
            plt.xlabel('Julian Date')
            plt.ylabel('Relative Flux')
            plt.show()
        # Return the results DataFrame
        return results_df

    def perform_e_ap_phot(self, filt,
        save = True,
        plot = True,
        display_apertures = False, # display the apertures and annuli on the reference image
        save_ref_star_coords = False, # save the star list in ra/dec coords for reference image
        title = None):
        '''
        Function to perform elliptical aperture photometry on the images (for windy or bouncy images)
        Not sure this will totally function right now, might need to be adjusted, should be close.
        '''
        # load images
        images = self._load_images(filt)
        if not images:
            raise FileNotFoundError(f"No images for filter {filt}")
        # read header and world coords of reference
        ref = self.ref_image or images[0]
        hdr = self._read_header(ref)
        wcs = WCS(hdr)
        coords = [wcs.all_pix2world(x, y, 0) for x, y in self.star_list_pix]
        # set default apertures from instance values
        a = self.a * self.ap_radius
        b = self.b * self.ap_radius
        a_in = a + self.gap
        a_out = a_in + 5
        b_in = b + self.gap
        b_out = b_in + 5
        if self.target_name is None:
            try:
                target_name = hdr['BLKNAME']
            except KeyError:
                target_name = hdr['OBJECT']
            print('Target Name:', target_name)
        else:
            target_name = self.target_name
            print('Target Name:', target_name)
        # Convert the target coordinates to degrees
        Simbad.add_votable_fields('pmra', 'pmdec')
        result = Simbad.query_object(target_name)
        target = SkyCoord(
            ra=result['ra'][0],
            dec=result['dec'][0],
            unit=(u.deg, u.deg),
            pm_ra_cosdec=result['pmra'][0]*u.mas/u.yr,
            pm_dec=result['pmdec'][0]*u.mas/u.yr,
            obstime=Time('J2000.0'),
            frame='icrs'
        )
        target = target.apply_space_motion(new_obstime=Time(datetime.now().isoformat()))
        ra = target.ra.deg
        dec = target.dec.deg
        print('Target RA:', ra)
        print('Target Dec:', dec)
        coords = [(ra, dec)] + coords
        all_results = []
        # Check if the reference star coordinates are provided
        # if ref_star_coords:
        #     print('Using reference star coordinates')
        #     star_list_ra_dec = pd.read_csv(ref_star_coords).values.tolist()
        # Convert the star list from pixel coordinates to RA/Dec coordinates
        # Initialize a list to store results for all files
        for im in images:
            data = fits.getdata(im)
            hdr = self._read_header(im)
            wcs = WCS(hdr)
            xy = [wcs.all_world2pix(ra, dec, 0) for ra, dec in coords]
            if save_ref_star_coords:
                star_list_saving = pd.DataFrame(coords, columns=['RA', 'Dec'])
                target_name = target_name.replace(' ', '_')
                star_list_saving.to_csv(self.file_dir + '/' + title + '_ref_star_coords.csv', index=False)
                print('Saving reference star coordinates to', title + '_ref_star_coords.csv')

            # Pull the exposure time from the header, convert to days
            # Get the Julian Date from the header and add the half exposure time to get the mid-exposure time
            exptime = hdr['EXPTIME']
            exptimedays = exptime/(24*3600)
            addjd = exptimedays/2
            jd = hdr['JD']
            file_results = {'file': im, 'Julian_Date': (jd+addjd)}

            #Make a 2D background model of the image
            # bkg_estimator = MedianBackground()
            # bkg = Background2D(data, (30, 30), filter_size=(3, 3), sigma_clip=SigmaClip(sigma=3), bkg_estimator=bkg_estimator)
            # new_data = data - bkg.background

            # Stats for sigma clipping
            sigclip = SigmaClip(sigma=3., maxiters=10)

            # Perform aperture photometry for each star
            for i, (x, y) in enumerate(xy):
                ap = EllipticalAperture((x, y), a=a, b=b, theta = self.theta)
                an = EllipticalAnnulus((x, y), a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out, theta = self.theta)
                # Begin Saturation Check
                mask = ap.to_mask(method='center')
                cutout = mask.cutout(data)
                ap_pixels = cutout[mask.data.astype(bool)]
                # Check for saturation
                if np.any(ap_pixels > 60000):
                    print(f"Saturated pixel found in {im}!")
                    skip_file = True
                bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                ap_stats = ApertureStats(data, ap)

                # Recentroid the aperture
                if bkg_stats.median > ap_stats.mean:
                    print("Backround is higher than the star, recentering")
                    an = CircularAnnulus((x, y), r_in=annulus_inner_radius+10, r_out=annulus_outer_radius+10)
                    bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                sub = data - bkg_stats.median
                x0, y0 = x, y
                try:
                    x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                    if np.isnan(x) or np.isnan(y):
                        raise ValueError
                    x, y = float(x), float(y)
                except Exception:
                    print(f"  → centroid failed for star {i} on image {im}, trying again")
                    x, y = x0, y0
                    ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                    x, y = ap_stats.centroid
                # try:
                #     x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                #     if np.isnan(x) or np.isnan(y):
                #         raise ValueError
                #     x, y = float(x), float(y)
                # except Exception:
                #     x, y = x0, y0

                aperture = EllipticalAperture((x, y), a= a, b=b, theta=self.theta)
                annulus_aperture = EllipticalAnnulus((x, y), a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out, theta = self.theta)

                # Perform aperture photometry
                phot_table = aperture_photometry(data, aperture)
                bkgstats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)

                # Calculate the background in the annulus
                bkg_mean = bkgstats.mean
                bkg_sum = bkg_mean * aperture.area

                # Subtract the pedestal from the background for the error calculation
                pedestal = hdr.get('PEDESTAL',0)
                bkg_mean_nopedestal = bkg_mean-pedestal
                bkg_sum_nopedestal = bkg_mean_nopedestal * aperture.area


                # Subtract the background from the aperture photometry
                source_sum = phot_table['aperture_sum'][0]*self.gain - bkg_sum*self.gain

                # Check if the  is negative and skip if it is
                if source_sum < 0:
                    print(f"  star {i}, pos=({x:.1f},{y:.1f}), "
                          f"ap_sum={phot_table['aperture_sum'][0]:.1f}, "
                          f"bkg_mean={bkg_mean:.1f}, bkg_sum={bkg_sum:.1f}, "
                          f"gain={self.gain}")
                    skip_file = True
                    break
                # if source_sum < 0:
                #     print(f"Skipping {im} due to negative , image should be inspected.")
                #     continue

                # if source_sum < 0:
                #     source_sum = 0.000001
                #     print(f" for {im} is negative, setting to 0.000001")

                # Error calculation (Poisson noise + background noise (not currently included) + read noise)
                error = np.sqrt((phot_table['aperture_sum'][0]-pedestal*aperture.area)*self.gain + (aperture.area/annulus_aperture.area)*bkg_sum_nopedestal*self.gain+aperture.area*self.rdnoise**2+ aperture.area**2/annulus_aperture.area*self.rdnoise**2)

                # Optionally turn into a magnitude (not used here but useful for reference)
                source_mag = -2.5 * np.log10(source_sum / exptime)
                source_mag_err = 1.0857 * error / source_sum

                # Store the results with dynamic column names
                file_results[f'star_{i}_x'] = x
                file_results[f'star_{i}_y'] = y
                file_results[f'star_{i}_flux'] = source_sum
                file_results[f'star_{i}_error'] = error
                #file_results[f'star_{i}_background'] = bkg_sum

            if display_apertures:
                plt.imshow(data, vmin=np.percentile(data, 5), vmax=np.percentile(data, 99), cmap='viridis')
                for (x, y) in self.star_list_pix:
                    aperture = EllipticalAperture((x, y), a=a, b=b, theta=self.theta)
                    annulus_aperture = EllipticalAnnulus((x, y), a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out, theta = self.theta)
                    aperture.plot(color='blue', lw=1.5)
                    annulus_aperture.plot(color='red', lw=1.5)
                plt.show()
            # Append the results for this file to the list of all results
            all_results.append(file_results)

        # Convert the results to a DataFrame for easy analysis
        results_df = pd.DataFrame(all_results)

        # Rename the columns to remove the 'star_0_' prefix for the first star
        # and replace it with 'target_' for clarity
        results_df.rename(
        columns=lambda c: c.replace("star_0_", "target_")
        if c.startswith("star_0_") else c,
        inplace=True)

        # add all the comparison stars together
        results_df['total_flux'] = results_df['star_1_flux'] + results_df['star_2_flux'] + results_df['star_3_flux'] + results_df['star_4_flux'] + results_df['star_5_flux']

        # calculate the relative flux of the target star
        results_df['target_rel_flux']=results_df['target_flux']/results_df['total_flux']

        # calculate the relative flux of the comparison stars
        for i in range(1,6):
            results_df[f'star_{i}_relflux'] = results_df[f'star_{i}_flux']/results_df['total_flux']

        # calculate the error on the total flux
        total_flux_err = np.sqrt(results_df['star_1_error']**2 + results_df['star_2_error']**2 + results_df['star_3_error']**2 + results_df['star_4_error']**2 + results_df['star_5_error']**2)

        #calculate the error on the relative flux
        results_df['target_relerror'] = (results_df['target_flux']/results_df['total_flux'])*(np.sqrt((results_df['target_error']/results_df['target_flux'])**2 + (total_flux_err/results_df['total_flux'])**2))

        # Normalize the relative flux
        mean_rel_flux, _, _, = sigma_clipped_stats(results_df['target_rel_flux'], sigma=  2.0)
        results_df['norm_target_rel_flux'] = results_df['target_rel_flux']/mean_rel_flux
        results_df['norm_target_rel_flux_error'] = results_df['target_relerror']/mean_rel_flux

        # Save the results to a CSV file
        if save:
            results_df.to_csv(self.file_dir + 'Results_' + filt + '.csv', index=False)

        if plot:
            #Plot the relative flux
            #plt.plot(results_df['Julian_Date'], results_df['target_rel_flux'], 'o')
            plt.errorbar(results_df['Julian_Date'], results_df['target_rel_flux'], yerr=results_df['target_relerror'], fmt='o')
            plt.xlabel('Julian Date')
            plt.ylabel('Relative Flux')
            plt.show()

        # Return the results DataFrame
        return results_df



    def find_fwhm(self, image, positions, size=30):
        """
        Computes approximate FWHM of stars by:
        1) subtracting a clipped background
        2) building a 1-D radial profile in a few bins
        3) finding the radius where profile crosses half-max
        """
        data = fits.getdata(image).astype(float)
        fwhm_list = []
        # make a sensible set of annular radii
        # (you only need one bin per pixel)
        radii = np.arange(size+1)

        for x0,y0 in positions:
            # centroid the star
            ap_stats = ApertureStats(data, CircularAperture((x0, y0), r=10), sigma_clip=SigmaClip(sigma=3))
            x0, y0 = ap_stats.centroid
            if np.isnan(x0) or np.isnan(y0):
                print(f"  FWHM centroid failed for star at {x0}, {y0} on image {image}, skipping")
                continue
            x0,y0 = int(x0), int(y0)
            # stamp boundary check
            if x0-size<0 or x0+size>=data.shape[1] or y0-size<0 or y0+size>=data.shape[0]:
                continue

            stamp = data[y0-size:y0+size+1, x0-size:x0+size+1]
            # mask out the core so bg estimate isn't biased
            yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
            core = ((yy-size)**2 + (xx-size)**2) < (size/4)**2
            _, med, _ = sigma_clipped_stats(stamp, mask=core, sigma=3, maxiters=5)
            stamp -= med

            # build a quick radial profile
            rp = RadialProfile(stamp, (size, size), radii, mask=None)
            profile = rp.profile
            radius  = rp.radius

            # compute half‐maximum
            half = profile.max() / 2.0
            # find index of the peak
            peak_idx = np.nanargmax(profile)
            # look for the first bin below half *after* the peak
            candidates = np.where((profile < half) & (radius > radius[peak_idx]))[0]
            if len(candidates) == 0:
                # no valid half‐max crossing on the far side
                continue
            i = candidates[0]
            # now linearly interpolate between bin i-1 and i
            p1, p2 = profile[i-1], profile[i]
            r1, r2 = radius[i-1],   radius[i]
            rhalf = r1 + (half - p1) * (r2-r1) / (p2-p1)
            # full width at half max:
            fwhm_list.append(2 * rhalf)

        return fwhm_list

    ###                                   ###
    ### Variable FWHM Photometry Function ###
    ###                                   ###     
    def perform_var_fwhm_phot(self, filt,
        save_ref_star_coords = False,
        display_apertures = False,
        save = True,
        plot = True, # plot the results
        errorshow = True,
        title = None,
        non_sidereal = False):
        '''
        Function to perform variable FWHM photometry on images
        for images with focus issues
        may also work for windy or bouncy images
        THIS IS THE WORKHORSE FUNCTION FOR PHOTOMETRY, IT WILL DO THE WORK OF THE OTHER TWO FUNCTIONS
        '''
        images = self._load_images(filt)
        if not images:
            raise FileNotFoundError(f"No images for filter {filt}")
        ref = self.ref_image or images[0]
        hdr = self._read_header(ref)
        wcs = WCS(hdr)
        # convert pix->world->pix for all
        coords = [wcs.all_pix2world(x, y, 0) for x, y in self.star_list_pix]
        # add the target star to the list
        if self.target_name is None:
            try:
                target_name = hdr['BLKNAME']
            except KeyError:
                target_name = hdr['OBJECT']
            print('Target Name:', target_name)
        else:
            target_name = self.target_name
            print('Target Name:', target_name)
        # Convert the target coordinates to degrees
        if not non_sidereal:
            Simbad.add_votable_fields('pmra', 'pmdec')
            result = Simbad.query_object(target_name)
            target = SkyCoord(
                ra=result['ra'][0],
                dec=result['dec'][0],
                unit=(u.deg, u.deg),
                pm_ra_cosdec=result['pmra'][0]*u.mas/u.yr,
                pm_dec=result['pmdec'][0]*u.mas/u.yr,
                obstime=Time('J2000.0'),
                frame='icrs'
            )
            target = target.apply_space_motion(new_obstime=Time(datetime.now().isoformat()))
            ra = target.ra.deg
            dec = target.dec.deg
            print('Target RA:', ra)
            print('Target Dec:', dec)
            coords = [(ra, dec)] + coords
        all_results = []

        # Loop over each image file
        for im in images:
            hdr = self._read_header(im)
            exptime = hdr['EXPTIME']
            jd = hdr['JD'] + exptime/(2*86400)
            if non_sidereal:
                non_sid_obj=Horizons(id=non_sid_id, location=non_sid_loc, epochs=jd)
                eph=non_sid_obj.ephemerides()
                ra=eph['RA'][0]
                dec=eph['DEC'][0]
                print('Target RA:', ra)
                print('Target Dec:', dec)
                coords = [(ra, dec)] + [wcs.all_pix2world(x, y, 0) for x, y in self.star_list_pix]
            data = fits.getdata(im)
            wcs = WCS(hdr)
            xy = [wcs.all_world2pix(ra, dec, 0) for ra, dec in coords]
            if save_ref_star_coords:
                star_list_saving = pd.DataFrame(coords, columns=['RA', 'Dec'])
                target_name = target_name.replace(' ', '_')
                star_list_saving.to_csv(self.file_dir + '/' + title + '_ref_star_coords.csv', index=False)
                print('Saving reference star coordinates to', title + '_ref_star_coords.csv')

            fwhm = self.find_fwhm(im, positions = xy, size = 30)
            medianfwhm = np.median(fwhm)

            aperture_radius = max(self.ap_radius, medianfwhm)  # Radius of the aperture
            annulus_inner_radius = max(self.an_inner, (medianfwhm*3))  # Inner radius of the annulus
            annulus_outer_radius = max(self.an_outer, annulus_inner_radius+5)  # Outer radius of the annulus
            aperture_radius = int(aperture_radius)
            annulus_inner_radius = int(annulus_inner_radius)
            annulus_outer_radius = int(annulus_outer_radius)

            if aperture_radius > 20:
                print(f"Warning: Aperture radius {aperture_radius} is larger than 20 pixels. Check the image {im}.")
                # aperture_radius = 20
                # annulus_inner_radius = 25
                # annulus_outer_radius = 30

            # Write aperture and annulus parameters to the header

            try:
              with fits.open(im, mode='update') as hdul:
                  hdr = hdul[1].header
                  hdr['AP_R']   = (aperture_radius,         'Variable FWHM aperture radius')
                  hdr['AN_R1']  = (annulus_inner_radius,    'Variable FWHM annulus inner radius')
                  hdr['AN_R2']  = (annulus_outer_radius,    'Variable FWHM annulus outer radius')
            except:
              with fits.open(im, mode='update') as hdul:
                  hdr = hdul[0].header
                  hdr['AP_R']   = (aperture_radius,         'Variable FWHM aperture radius')
                  hdr['AN_R1']  = (annulus_inner_radius,    'Variable FWHM annulus inner radius')
                  hdr['AN_R2']  = (annulus_outer_radius,    'Variable FWHM annulus outer radius')
            #    hdul.flush()   # writes your changes back to `im` in place

            file_results = {'file': im, 'Julian_Date': (jd)}

            # Stats for sigma clipping
            sigclip = SigmaClip(sigma=3., maxiters=10)
            skip_file = False
            # Begin WCS Check:
            try:
                header=self._read_header(im)
                wcs = WCS(header)
                # If WCS has no celestial axes, this will fail
                if not wcs.has_celestial:
                    raise ValueError("No celestial WCS")
                # Try a world → pixel transform as a functional test
                x1, y1 = wcs.all_world2pix(ra,dec,0)
                # Catch NaNs
                if not (x1 == x1 and y1 == y1):
                    raise ValueError("NaN pixel coordinates")
            except Exception:
                print("WCS Failure; skipping file: ", im)
                skip_file=True
                continue
            # Perform aperture photometry for each star
            for i, (x, y) in enumerate(xy):
                ap = CircularAperture((x, y), r=aperture_radius)
                an = CircularAnnulus((x, y), r_in=annulus_inner_radius, r_out=annulus_outer_radius)


                # Begin Saturation Check
                mask = ap.to_mask(method='center')
                cutout = mask.cutout(data)
                ap_pixels = cutout[mask.data.astype(bool)]
                # Check for saturation
                if np.any(ap_pixels > 60000):
                    print(f"Saturated pixel found in {im}!")
                    skip_file = True


                bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                #ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                ap_stats = ApertureStats(data, ap)

                # Recentroid the aperture
                if bkg_stats.median > ap_stats.mean:
                    print("Backround is higher than the star, recentering")
                    an = CircularAnnulus((x, y), r_in=annulus_inner_radius+10, r_out=annulus_outer_radius+10)
                    bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                sub = data - bkg_stats.median
                x0, y0 = x, y
                try:
                    x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                    if np.isnan(x) or np.isnan(y):
                        raise ValueError
                    x, y = float(x), float(y)
                except Exception:
                    print(f"  → centroid failed for star {i} on image {im}, trying again")
                    x, y = x0, y0
                    ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                    x, y = ap_stats.centroid
                # try:
                #     x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                #     if np.isnan(x) or np.isnan(y):
                #         raise ValueError
                #     x, y = float(x), float(y)
                # except Exception:
                #     x, y = x0, y0


                aperture = CircularAperture((x, y), r=aperture_radius)
                annulus_aperture = CircularAnnulus((x, y), r_in=annulus_inner_radius, r_out=annulus_outer_radius)

                # Perform aperture photometry
                phot_table = aperture_photometry(data, aperture)
                bkgstats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)

                # Calculate the background in the annulus
                bkg_mean = bkgstats.mean
                bkg_sum = bkg_mean * aperture.area

                # Subtract the pedestal from the background for the error calculation
                pedestal = hdr.get('PEDESTAL',0)
                bkg_mean_nopedestal = bkg_mean-pedestal
                bkg_sum_nopedestal = bkg_mean_nopedestal * aperture.area


                # Subtract the background from the aperture photometry
                source_sum = phot_table['aperture_sum'][0]*self.gain - bkg_sum*self.gain

                # Check if the  is negative and skip if it is
                if source_sum < 0:
                    print(f"  star {i}, pos=({x:.1f},{y:.1f}), "
                          f"ap_sum={phot_table['aperture_sum'][0]:.1f}, "
                          f"bkg_mean={bkg_mean:.1f}, bkg_sum={bkg_sum:.1f}, "
                          f"gain={self.gain}")
                    skip_file = True
                    break
                # if source_sum < 0:
                #     print(f"Skipping {im} due to negative , image should be inspected.")
                #     skip_file = True
                #     break

                # if source_sum < 0:
                #     source_sum = 0.000001
                #     print(f" for {im} is negative, setting to 0.000001")

                # Error calculation (Poisson noise + background noise + read noise)
                error = np.sqrt((phot_table['aperture_sum'][0]-pedestal*aperture.area)*self.gain + ((aperture.area)/annulus_aperture.area)*bkg_sum_nopedestal*self.gain + aperture.area*self.rdnoise**2 + aperture.area**2/annulus_aperture.area*self.rdnoise**2)

                # Optionally turn into a magnitude (not used here but useful for reference)
                source_mag = -2.5 * np.log10(source_sum / exptime)
                source_mag_err = 1.0857 * error / source_sum

                # Store the results with dynamic column names
                file_results[f'star_{i}_x'] = x
                file_results[f'star_{i}_y'] = y
                file_results[f'star_{i}_flux'] = source_sum
                file_results[f'star_{i}_error'] = error
                file_results[f'star_{i}_background'] = bkg_sum

            # Optionally, display the image with the apertures and annuli (set flag to True)
            # This is useful for checking the positions of the stars are correct
            if display_apertures:
                # Make a cutout plot of stars

                # set the figure size to be 10x10 inches
                plt.figure(figsize=(10, 10))
                plt.imshow(data, vmin=np.percentile(data, 5), vmax=np.percentile(data, 99), cmap='viridis')
                for (x, y) in self.star_list_pix:
                    aperture = CircularAperture((x, y), r=aperture_radius)
                    annulus_aperture = CircularAnnulus((x, y), r_in=annulus_inner_radius, r_out=annulus_outer_radius)
                    aperture.plot(color='blue', lw=1.5)
                    annulus_aperture.plot(color='red', lw=1.5)
                plt.show()

            # Append the results for this file to the list of all results
            if skip_file:
                print(f"Skipping {im} due to negative , image should be inspected.")
                continue
            else:
                # Append the results for this file to the list of all results
                all_results.append(file_results)

        # Convert the results to a DataFrame for easy analysis
        results_df = pd.DataFrame(all_results)

        # Rename the columns to remove the 'star_0_' prefix for the first star
        # and replace it with 'target_' for clarity
        results_df.rename(
        columns=lambda c: c.replace("star_0_", "target_")
        if c.startswith("star_0_") else c,
        inplace=True
        )

        # add all the comparison stars together
        results_df['total_flux'] = results_df['star_1_flux'] + results_df['star_2_flux'] + results_df['star_3_flux'] + results_df['star_4_flux']+ results_df['star_5_flux']

        # calculate the relative flux of the target star
        results_df['target_rel_flux']=results_df['target_flux']/results_df['total_flux']

        # calculate the relative flux of the comparison stars
        for i in range(1,6):
            results_df[f'star_{i}_relflux'] = results_df[f'star_{i}_flux']/results_df['total_flux']

        # calculate the error on the total flux
        total_flux_err = np.sqrt(results_df['star_1_error']**2 + results_df['star_2_error']**2 + results_df['star_3_error']**2 + results_df['star_4_error']**2 + results_df['star_5_error']**2)

        #calculate the error on the relative flux
        results_df['target_relerror'] = (results_df['target_flux']/results_df['total_flux'])*(np.sqrt((results_df['target_error']/results_df['target_flux'])**2 + (total_flux_err/results_df['total_flux'])**2))

        # Normalize the relative flux
        mean_rel_flux, _, _, = sigma_clipped_stats(results_df['target_rel_flux'], sigma=  2.0)
        results_df['norm_target_rel_flux'] = results_df['target_rel_flux']/mean_rel_flux
        results_df['norm_target_rel_flux_error'] = results_df['target_relerror']/mean_rel_flux

        # Save the results to a CSV file
        if save:
            results_df.to_csv(self.file_dir + "/"  + title + 'Results_' + filt + '.csv', index=False)

        if plot:
            #Plot the relative flux
            #plt.plot(results_df['Julian_Date'], results_df['target_rel_flux'], 'o')
            if errorshow:
                plt.errorbar(results_df['Julian_Date'], results_df['target_rel_flux'], yerr=results_df['target_relerror'], fmt='o')
            else:
                plt.scatter(results_df['Julian_Date'], results_df['target_rel_flux'])
            plt.xlabel('Julian Date')
            plt.ylabel('Relative Flux')
            plt.show()

        # Return the results DataFrame
        return results_df

    def calc_ap_ratio(self, filt, larger_ap, mean=True, median=False, title = None):
        '''
        Calculate the ratio of fluxes between a larger and smaller aperture for a given filter.
        Neccessary if you are going to convert from relative flux to magnitudes.
        '''
        # Try to see if AP_R, AN_R1, and AN_R2 are in the header of self
        df_small = self.perform_phot(filt, save=False, plot=False, use_hdr_ap=True, large_ap=False, title = title)
        df_large = self.perform_phot(filt, save=False, plot=False, use_hdr_ap=True, large_ap=True, title = title)

        df = df_small.merge(df_large, on='file', suffixes=('_small','_large'))
        ratio_cols = [f'star_{i}_flux_ratio' for i in range(1,6)]
        for i in range(1,6):
            df[f'star_{i}_flux_ratio'] = (
                df[f'star_{i}_flux_large'] / df[f'star_{i}_flux_small']
            )
            df[f'star_{i}_flux_ratio_err'] = np.sqrt(
                (df[f'star_{i}_error_large'] / df[f'star_{i}_flux_large'])**2 +
                (df[f'star_{i}_error_small'] / df[f'star_{i}_flux_small'])**2
            )

        if mean:
            df['mean_ratio'] = df[ratio_cols].mean(axis=1)
            df['mean_error'] = df[ratio_cols].std(axis=1)
            try:
              for fn, scale in zip(df['file'], df['mean_ratio']):
                  fits.setval(fn, ext=1, keyword='SCALE', value=scale,
                              comment='Photometric scale factor')
            except Exception:
              for fn, scale in zip(df['file'], df['mean_ratio']):
                  fits.setval(fn, ext=0, keyword='SCALE', value=scale,
                              comment='Photometric scale factor')
            try:
              for fn, err in zip(df['file'], df['mean_error']):
                  fits.setval(fn, ext=1, keyword='SC_ERR', value=err,
                              comment='Scale error (std of ratios)')
            except Exception:
              for fn, err in zip(df['file'], df['mean_error']):
                  fits.setval(fn, ext=0, keyword='SC_ERR', value=err,
                              comment='Scale error (std of ratios)')

        if median:
            df['median_ratio'] = df[ratio_cols].median(axis=1)
            df['median_error'] = df[ratio_cols].std(axis=1)
            try:
              for fn, scale in zip(df['file'], df['median_ratio']):
                  fits.setval(fn, ext=1, keyword='SCALE', value=scale,
                              comment='Photometric scale factor')
            except Exception:
              for fn, scale in zip(df['file'], df['median_ratio']):
                  fits.setval(fn, ext=0, keyword='SCALE', value=scale,
                              comment='Photometric scale factor')
            try:
              for fn, err in zip(df['file'], df['median_error']):
                  fits.setval(fn, ext=1, keyword='SC_ERR', value=err,
                              comment='Scale error (std of ratios)')
            except Exception:
              for fn, err in zip(df['file'], df['median_error']):
                  fits.setval(fn, ext=0, keyword='SC_ERR', value=err,
                              comment='Scale error (std of ratios)')

        return df


    def find_mags(self, filt, save = True, save_ref_star_coords = False, display_apertures = False, HJD = False, use_hdr_ap = True, title = None):
        '''
        Find the apparent magnitudes of the stars in the image
        Can only be done after you have done the photometry and the aperture ratio has been calculated
        '''
        images = self._load_images(filt)
        if not images:
            raise FileNotFoundError(f"No images for filter {filt}")
        ref = self.ref_image or images[0]
        hdr = self._read_header(ref)
        wcs = WCS(hdr)
        # convert pix->world->pix for all
        coords = [wcs.all_pix2world(x, y, 0) for x, y in self.star_list_pix]
        # add the target star to the list
        if self.target_name is None:
            try:
                target_name = hdr['BLKNAME']
            except KeyError:
                target_name = hdr['OBJECT']
            print('Target Name:', target_name)
        else:
            target_name = self.target_name
            print('Target Name:', target_name)
        # Convert the target coordinates to degrees
        Simbad.add_votable_fields('pmra', 'pmdec')
        result = Simbad.query_object(target_name)
        target = SkyCoord(
            ra=result['ra'][0],
            dec=result['dec'][0],
            unit=(u.deg, u.deg),
            pm_ra_cosdec=result['pmra'][0]*u.mas/u.yr,
            pm_dec=result['pmdec'][0]*u.mas/u.yr,
            obstime=Time('J2000.0'),
            frame='icrs'
        )
        target = target.apply_space_motion(new_obstime=Time(datetime.now().isoformat()))
        ra = target.ra.deg
        dec = target.dec.deg
        print('Target RA:', ra)
        print('Target Dec:', dec)
        coords = [(ra, dec)] + coords
        all_results = []
        for im in images:
            data = fits.getdata(im)
            hdr = self._read_header(im)
            wcs = WCS(hdr)
            xy = [wcs.all_world2pix(ra, dec, 0) for ra, dec in coords]
            if save_ref_star_coords:
                star_list_saving = pd.DataFrame(coords, columns=['RA', 'Dec'])
                target_name = target_name.replace(' ', '_')
                star_list_saving.to_csv(self.file_dir + '/' + target_name + '_ref_star_coords.csv', index=False)
                print('Saving reference star coordinates to', + target_name + '_ref_star_coords.csv')
            if HJD:
                exptime = hdr['EXPTIME']
                jd = hdr['HJD'] + exptime/(2*86400)
            else:
                exptime = hdr['EXPTIME']
                jd = hdr['JD'] + exptime/(2*86400)
            file_results = {'file': im, 'Julian_Date': jd}
            sigclip = SigmaClip(sigma=3., maxiters=10)
            skip_file = False
            # Begin WCS Check:
            try:
                header=self._read_header(im)
                wcs = WCS(header)
                # If WCS has no celestial axes, this will fail
                if not wcs.has_celestial:
                    raise ValueError("No celestial WCS")
                # Try a world → pixel transform as a functional test
                x1, y1 = wcs.all_world2pix(ra,dec,0)
                # Catch NaNs
                if not (x1 == x1 and y1 == y1):
                    raise ValueError("NaN pixel coordinates")
            except Exception:
                print("WCS Failure; skipping file: ", im)
                skip_file=True
                continue
            for i, (x, y) in enumerate(xy):
                if use_hdr_ap:
                    ap_r = hdr['AP_R']
                    an_i = hdr['AN_R1']
                    an_o = hdr['AN_R2']
                else:
                    ap_r = self.ap_radius
                    an_i = self.an_inner
                    an_o = self.an_outer
                ap = CircularAperture((x, y), r=ap_r)
                an = CircularAnnulus((x, y), r_in=an_i, r_out=an_o)
                # Begin Saturation Check
                mask = ap.to_mask(method='center')
                cutout = mask.cutout(data)
                ap_pixels = cutout[mask.data.astype(bool)]
                # Check for saturation
                if np.any(ap_pixels > 60000):
                    print(f"Saturated pixel found in {im}!")
                    skip_file = True
                bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                #ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                ap_stats = ApertureStats(data, ap)

                # Recentroid the aperture
                if bkg_stats.median > ap_stats.mean:
                    print("Backround too high, recentroiding")
                    an = CircularAnnulus((x, y), r_in=an_i+10, r_out=an_o+10)
                    bkg_stats = ApertureStats(data, an, sigma_clip=sigclip)
                sub = data - bkg_stats.median
                x0, y0 = x, y
                try:
                    x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                    if np.isnan(x) or np.isnan(y):
                        raise ValueError
                    x, y = float(x), float(y)
                except Exception:
                    print(f"  → centroid failed for star {i} on image {im}, trying again")
                    x, y = x0, y0
                    ap_stats = ApertureStats(data, ap, local_bkg=bkg_stats.median)
                    x, y = ap_stats.centroid
                # try:
                #     x, y = centroid_sources(sub, x, y, box_size=(25,25), centroid_func=centroid_2dg)
                #     if np.isnan(x) or np.isnan(y):
                #         raise ValueError
                #     x, y = float(x), float(y)
                # except Exception:
                #     x, y = x0, y0


                aperture = CircularAperture((x, y), r=ap_r)
                annulus_aperture = CircularAnnulus((x, y), r_in=an_i, r_out=an_o)

                # Perform aperture photometry
                phot_table = aperture_photometry(data, aperture)
                bkgstats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)

                # Calculate the background in the annulus
                bkg_mean = bkgstats.mean
                bkg_sum = bkg_mean * aperture.area

                # Subtract the pedestal from the background for the error calculation
                pedestal = hdr.get('PEDESTAL',0)
                bkg_mean_nopedestal = bkg_mean-pedestal
                bkg_sum_nopedestal = bkg_mean_nopedestal * aperture.area

                # Subtract the background from the aperture photometry
                source_sum = phot_table['aperture_sum'][0]*self.gain - bkg_sum*self.gain

                # Check if the  is negative and skip if it is
                if source_sum < 0:
                    print(f"  star {i}, pos=({x:.1f},{y:.1f}), "
                          f"ap_sum={phot_table['aperture_sum'][0]:.1f}, "
                          f"bkg_mean={bkg_mean:.1f}, bkg_sum={bkg_sum:.1f}, "
                          f"gain={self.gain}")
                    skip_file = True
                    break
                # if source_sum < 0:
                #     print(f"Skipping {im} due to negative , image should be inspected.")
                #     skip_file = True
                #     break

                # if source_sum < 0:
                #     source_sum = 0.000001
                #     print(f"Source sum for {im} is negative, setting to 0.000001")

                # Error calculation (Poisson noise + background noise + read noise)
                error = np.sqrt((phot_table['aperture_sum'][0]-pedestal*aperture.area)*self.gain + ((aperture.area)/annulus_aperture.area)*bkg_sum_nopedestal*self.gain + aperture.area*self.rdnoise**2 + aperture.area**2/annulus_aperture.area*self.rdnoise**2)

                # Optionally turn into a magnitude
                hdr = self._read_header(im)
                try:
                    scale = hdr['SCALE']
                except KeyError:
                #    If the scale is not found in the header, set it to 1
                    scale = 1.0
                    print('Scale not found in header, using default value of 1.0')
                try:
                    zpmag = hdr['ZMAG']
                    zpmag_err = hdr['ZMAGERR']
                    source_mag = -2.5 * np.log10(scale*source_sum / exptime)
                    source_mag_cor = source_mag + zpmag
                    source_mag_err = 1.0857 * error / source_sum
                    source_mag_cor_err = np.sqrt(source_mag_err**2 + zpmag_err**2)
                    file_results[f'star_{i}_mag'] = source_mag_cor
                    file_results[f'star_{i}_mag_error'] = source_mag_cor_err
                except KeyError:
                    # If the zero point magnitude or zero point error is not found in the header, skip this file
                    print('Zero point magnitude or error not found in header of ', im)
                # Store the results with dynamic column names
                file_results[f'star_{i}_x'] = x
                file_results[f'star_{i}_y'] = y
                file_results[f'star_{i}_flux'] = source_sum
                file_results[f'star_{i}_error'] = error
                file_results[f'star_{i}_background'] = bkg_sum
                # Append the results for this file to the list of all results


            # Optionally, display the image with the apertures and annuli (set flag to True)
            # This is useful for checking the positions of the stars are correct
            if display_apertures:
                plt.imshow(data, vmin=np.percentile(data, 5), vmax=np.percentile(data, 99), cmap='viridis')
                for (x, y) in self.star_list_pix:
                    aperture = CircularAperture((x, y), r=ap_r)
                    annulus_aperture = CircularAnnulus((x, y), r_in=an_i, r_out=an_o)
                    aperture.plot(color='blue', lw=1.5)
                    annulus_aperture.plot(color='red', lw=1.5)
                plt.show()

            # Append the results for this file to the list of all results
            all_results.append(file_results)

        # Convert the results to a DataFrame for easy analysis
        results_df = pd.DataFrame(all_results)
        # Rename the columns to remove the 'star_0_' prefix for the first star

        # and replace it with 'target_' for clarity
        results_df.rename(
        columns=lambda c: c.replace("star_0_", "target_")
        if c.startswith("star_0_") else c,
        inplace=True
        )

        # add all the comparison stars together
        results_df['total_flux'] = results_df['star_1_flux'] + results_df['star_2_flux'] + results_df['star_3_flux'] + results_df['star_4_flux'] + results_df['star_5_flux']

        # calculate the relative flux of the target star
        results_df['target_rel_flux']=results_df['target_flux']/results_df['total_flux']

        # calculate the relative flux of the comparison stars
        for i in range(1,6):
            results_df[f'star_{i}_relflux'] = results_df[f'star_{i}_flux']/results_df['total_flux']

        # calculate the error on the total flux
        total_flux_err = np.sqrt(results_df['star_1_error']**2 + results_df['star_2_error']**2 + results_df['star_3_error']**2 + results_df['star_4_error']**2 + results_df['star_5_error']**2)

        #calculate the error on the relative flux
        results_df['target_relerror'] = (results_df['target_flux']/results_df['total_flux'])*(np.sqrt((results_df['target_error']/results_df['target_flux'])**2 + (total_flux_err/results_df['total_flux'])**2))

        # Normalize the relative flux
        mean_rel_flux, _, _, = sigma_clipped_stats(results_df['target_rel_flux'], sigma=  2.0)
        results_df['norm_target_rel_flux'] = results_df['target_rel_flux']/mean_rel_flux
        results_df['norm_target_rel_flux_error'] = results_df['target_relerror']/mean_rel_flux

        # Save the results to a CSV file
        if save:
            results_df.to_csv(self.file_dir + '/' + title + 'Results_Cal_' + filt + '.csv', index=False)



##################################
### Light Curve Plotting Class ###
##################################

class LightCurvePlotter:

    @staticmethod
    def plot_relative(file_dir, filt, comp_stars=False, title=None, phase = False, period = None, phase_zero = None, save_path = "", errorshow = True):
        # If filters are not provided, set it to ['g', 'r', 'i']
        if phase == True:
            comp_stars = False
        if filt is None:
            print('No filters provided. Using default filters: g, r, i')
            filt = ['g', 'r', 'i']
        else:
            print('Using filters:', filt)

        # Check if g is in the filters list
        try:
            lc_g = pd.read_csv(file_dir + '/' + title + 'Results_g.csv')
            if phase:
                if phase_zero is None:
                    print("phase=True but period or phase_zero is None; skipping phase.")
                    phase = False
                if period is None:
                    print("phase=True but period is None; skipping phase.")
                    phase = False
                lc_g['phase'] = (lc_g['Julian_Date'] - phase_zero) / period % 1
                # sort by phase
                lc_g = lc_g.sort_values(by=['phase'])
                plt.scatter(lc_g['phase'], lc_g['norm_target_rel_flux'], label='g', color='green')
                if errorshow:
                    plt.errorbar(lc_g['phase'], lc_g['norm_target_rel_flux'], yerr=lc_g['norm_target_rel_flux_error'], fmt='o', color='green')
                plt.xlabel('Phase')
            else:
                lc_g = lc_g.sort_values(by=['Julian_Date'])
                if errorshow:
                    plt.errorbar(lc_g['Julian_Date'], lc_g['norm_target_rel_flux'], yerr=lc_g['norm_target_rel_flux_error'], fmt='o', label='g', color='green')
                else:
                    plt.scatter(lc_g['Julian_Date'], lc_g['norm_target_rel_flux'], label='g', color='green')
                plt.plot(lc_g['Julian_Date'], lc_g['norm_target_rel_flux'], color='green')
                plt.xlabel('Julian Date')
        except FileNotFoundError:
            print('No g filter data found. Skipping g filter.')
        try:
            lc_r = pd.read_csv(file_dir + '/' + title + 'Results_r.csv')
            print("Exact columns:", lc_r.columns.tolist())
            if phase:
                if phase_zero is None:
                    print("phase=True but period or phase_zero is None; skipping phase.")
                    phase = False
                if period is None:
                    print("phase=True but period is None; skipping phase.")
                    phase = False
                lc_r['phase'] = (lc_r['Julian_Date'] - phase_zero) / period % 1
                # sort by phase
                lc_r = lc_r.sort_values(by=['phase'])
                plt.scatter(lc_r['phase'], lc_r['norm_target_rel_flux'], label='r', color='red')
                if errorshow:
                    plt.errorbar(lc_r['phase'], lc_r['norm_target_rel_flux'], yerr=lc_r['norm_target_rel_flux_error'], fmt='o', color='red')
                plt.xlabel('Phase')
            else:
                lc_r = lc_r.sort_values(by=['Julian_Date'])
                if errorshow:
                    plt.errorbar(lc_r['Julian_Date'], lc_r['norm_target_rel_flux'], yerr=lc_r['norm_target_rel_flux_error'], fmt='o', label='r', color='red')
                else:
                    plt.scatter(lc_r['Julian_Date'], lc_r['norm_target_rel_flux'], label='r', color='red')
                plt.plot(lc_r['Julian_Date'], lc_r['norm_target_rel_flux'], color='red')
                plt.xlabel('Julian Date')
        except FileNotFoundError:
            print('No r filter data found. Skipping r filter.')
        try:
            lc_i = pd.read_csv(file_dir + '/' + title + 'Results_i.csv')
            if phase:
                if phase_zero is None:
                    print("phase=True but period or phase_zero is None; skipping phase.")
                    phase = False
                if period is None:
                    print("phase=True but period is None; skipping phase.")
                    phase = False
                lc_i['phase'] = (lc_i['Julian_Date'] - phase_zero) / period % 1
                # sort by phase
                lc_i = lc_i.sort_values(by=['phase'])
                plt.scatter(lc_i['phase'], lc_i['norm_target_rel_flux'], label='i', color='purple')
                if errorshow:
                    plt.errorbar(lc_i['phase'], lc_i['norm_target_rel_flux'], yerr=lc_i['norm_target_rel_flux_error'], fmt='o', color='purple')
                plt.xlabel('Phase')
            else:
                lc_i = lc_i.sort_values(by=['Julian_Date'])
                if errorshow:
                    plt.errorbar(lc_i['Julian_Date'], lc_i['norm_target_rel_flux'], yerr=lc_i['norm_target_rel_flux_error'], fmt='o', label='i', color='purple')
                else:
                    plt.scatter(lc_i['Julian_Date'], lc_i['norm_target_rel_flux'], label='i', color='purple')
                plt.plot(lc_i['Julian_Date'], lc_i['norm_target_rel_flux'], color='purple')
                plt.xlabel('Julian Date')
        except FileNotFoundError:
            print('No i filter data found. Skipping i filter.')

        # If comp_stars is True, plot the comparison stars
        if comp_stars:
            try:
                lc_g = pd.read_csv(file_dir + '/' + title + 'Results_g.csv')
                lc_g = lc_g.sort_values(by=['Julian_Date'])
                plt.scatter(lc_g['Julian_Date'], lc_g['star_1_relflux']/np.mean(lc_g['star_1_relflux']), alpha=0.2, color='C0')
                plt.scatter(lc_g['Julian_Date'], lc_g['star_2_relflux']/np.mean(lc_g['star_2_relflux']), alpha=0.2, color='C1')
                plt.scatter(lc_g['Julian_Date'], lc_g['star_3_relflux']/np.mean(lc_g['star_3_relflux']), alpha=0.2, color='C2')
                plt.scatter(lc_g['Julian_Date'], lc_g['star_4_relflux']/np.mean(lc_g['star_4_relflux']), alpha=0.2, color='C3')
                plt.scatter(lc_g['Julian_Date'], lc_g['star_5_relflux']/np.mean(lc_g['star_5_relflux']), alpha=0.2, color='C4')
            except FileNotFoundError:
                print('No g filter data found. Skipping g filter.')
            try:
                lc_r = pd.read_csv(file_dir + '/' + title + 'Results_r.csv')
                lc_r = lc_r.sort_values(by=['Julian_Date'])
                plt.scatter(lc_r['Julian_Date'], lc_r['star_1_relflux']/np.mean(lc_r['star_1_relflux']), alpha=0.2, color='C0')
                plt.scatter(lc_r['Julian_Date'], lc_r['star_2_relflux']/np.mean(lc_r['star_2_relflux']), alpha=0.2, color='C1')
                plt.scatter(lc_r['Julian_Date'], lc_r['star_3_relflux']/np.mean(lc_r['star_3_relflux']), alpha=0.2, color='C2')
                plt.scatter(lc_r['Julian_Date'], lc_r['star_4_relflux']/np.mean(lc_r['star_4_relflux']), alpha=0.2, color='C3')
                plt.scatter(lc_r['Julian_Date'], lc_r['star_5_relflux']/np.mean(lc_r['star_5_relflux']), alpha=0.2, color='C4')
            except FileNotFoundError:
                print('No r filter data found. Skipping r filter.')
            try:
                lc_i = pd.read_csv(file_dir + '/' + title + 'Results_i.csv')
                lc_i = lc_i.sort_values(by=['Julian_Date'])
                plt.scatter(lc_i['Julian_Date'], lc_i['star_1_relflux']/np.mean(lc_i['star_1_relflux']), alpha=0.2, color='C0')
                plt.scatter(lc_i['Julian_Date'], lc_i['star_2_relflux']/np.mean(lc_i['star_2_relflux']), alpha=0.2, color='C1')
                plt.scatter(lc_i['Julian_Date'], lc_i['star_3_relflux']/np.mean(lc_i['star_3_relflux']), alpha=0.2, color='C2')
                plt.scatter(lc_i['Julian_Date'], lc_i['star_4_relflux']/np.mean(lc_i['star_4_relflux']), alpha=0.2, color='C3')
                plt.scatter(lc_i['Julian_Date'], lc_i['star_5_relflux']/np.mean(lc_i['star_5_relflux']), alpha=0.2, color='C4')
            except FileNotFoundError:
                print('No i filter data found. Skipping i filter.')
        plt.ylabel('Normalized Relative Flux')
        plt.legend()
        # convert ut to JD
        # ut_start = '2025-05-16T04:30:00'
        # ut_end = '2025-05-16T08:42:30'

        # jd_start = Time(ut_start, format='isot', scale='utc').jd
        # jd_end = Time(ut_end, format='isot', scale='utc').jd
        # # Plot the vertical lines
        # plt.axvline(x=jd_start, color='black', linestyle='--', label='Start Time')
        # plt.axvline(x=jd_end, color='black', linestyle='--', label='End Time')
        # # fill the area between the lines
        # plt.fill_betweenx([0, 1.5], jd_start, jd_end, color='blue', alpha=0.2)
        # Give labels to the axes and a title
        if title is None:
            plt.title('Light Curve')
            plt.show()
        else:
            plt.title(title)
        title = title.replace(' ', '_')
        if phase:
            plt.savefig(save_path + "/"  + title +"_" + "_phased.png")
            plt.show()
        else:
            plt.savefig(save_path + "/" + title +"_" + "_unphased.png")
            plt.show()



    @staticmethod
    def plot_absolute(
        file_dir: str,
        filt=None,
        title: str = None,
        phase: bool = False,
        period: float = None,
        phase_zero: float = None
    ):
        """
        Scatter‐plots the apparent magnitudes for all filters together,
        optionally phase‐folded.
        """
        # Title default
        if title is None:
            title = 'Absolute Light Curve'

        # Which bands?
        bands = filt if filt is not None else ['g','r','i']
        colors = {'g':'green','r':'red','i':'purple'}

        # If phase‐fold requested without all params, disable it
        if phase and (period is None or phase_zero is None):
            print("phase=True but missing period/phase_zero; plotting un‐folded.")
            phase = False

        xlabel = 'Phase' if phase else 'Julian Date'

        plt.figure(figsize=(8,5))
        for band in bands:
            csv_path = os.path.join(file_dir + '/' + title + f'Results_Cal_{band}.csv')
            if not os.path.exists(csv_path):
                print(f'No {band} data at {csv_path}, skipping.')
                continue

            df = pd.read_csv(csv_path)
            # must have these columns:
            if any(col not in df.columns for col in ('Julian_Date','target_mag','target_mag_error')):
                print(f"Missing columns in {csv_path}; need Julian_Date, target_mag, target_mag_error.")
                continue

            # build a small working copy
            df2 = df[['Julian_Date','target_mag','target_mag_error']].copy()
            df2 = df2.dropna(subset=['target_mag', 'target_mag_error'])

            if phase:
                df2['phase'] = ((df2['Julian_Date'] - phase_zero) / period) % 1
                df2.sort_values('phase', inplace=True)
                x = df2['phase']
            else:
                df2.sort_values('Julian_Date', inplace=True)
                x = df2['Julian_Date']

            # scatter the points
            plt.scatter(x, df2['target_mag'],
                        label=band,
                        color=colors.get(band,'black'),
                        alpha=0.8)
            # add error bars
            plt.errorbar(x, df2['target_mag'],
                        yerr=df2['target_mag_error'],
                        fmt='none',
                        ecolor=colors.get(band,'black'),
                        alpha=0.6)

        plt.xlabel(xlabel)
        plt.ylabel('Apparent Magnitude')
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.legend(title="Filter")
        plt.tight_layout()
        plt.show()


# class ZPmagrunner:
#     """
#     Encapsulates calling an external ZP magnitude script across filter subfolders.
#     """
#     def __init__(self,
#                  base_dir: str,
#                  script_path: str,
#                  flags='--writeAVG'):
#         self.base_dir = base_dir
#         self.script = script_path
#         # normalize flags into a list of strings
#         if isinstance(flags, str):
#             # split a single string into ['--foo','--bar']
#             self.flags = shlex.split(flags)
#         else:
#             # assume it's already a list
#             self.flags = flags

#     def run(self, filt):
#         orig = os.getcwd()
#         target_dir = os.path.join(self.base_dir, filt)
#         print('Processing folder:', target_dir)
#         # build the command by unpacking the flags list
#         cmd = ['python', self.script, '.'] + self.flags
#         subprocess.run(cmd, cwd=target_dir, check=True)
#         os.chdir(orig)
