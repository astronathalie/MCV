import os
from astropy.io import fits
from astropy.wcs import WCS
import glob
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from astroquery.sdss import SDSS
from astropy import units as u
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder
from photutils.aperture import SkyCircularAperture
import random as random
import math
from scipy.optimize import curve_fit

class SDSS_catalog:
    def __init__(self, PS = False):
        self.PS = PS
        pass

    def get_image_bounds(self, wcs):
        # Get the bounds of the image in sky coordinates
        naxis1 = wcs.array_shape[1]
        naxis2 = wcs.array_shape[0]
        
        # Get all four corners of the image
        corners = [(0, 0), (0, naxis2), (naxis1, 0), (naxis1, naxis2)]
        ra, dec = wcs.all_pix2world(corners, 0).T

        # Get the min and max of the RA and Dec
        # Handle RA wrapping
        ra_min, ra_max = min(ra), max(ra)

        # Check if the range spans the 0/360 boundary
        if ra_max - ra_min > 180:
            # Normalize RA values to the range 0 to 360
            ra = [(r + 360) if r < 180 else r for r in ra]
            ra_min, ra_max = min(ra), max(ra)

        # Return RA bounds in the standard range (0 to 360)     
        ra_min = ra_min % 360
        ra_max = ra_max % 360
        dec_min, dec_max = min(dec), max(dec)

        return ra_min, ra_max, dec_min, dec_max

    def get_catalog(self, wcs, filter, mag_range=(12, 16), num_sources=200, mag_limit=20):
        # Get the bounds of the image in sky coordinates
        PS = self.PS
        ra_min, ra_max, dec_min, dec_max = self.get_image_bounds(wcs)

        # Get the exposure time
        #self.header = fits.getheader(file)
        #exp_time = self.header['EXPTIME']

        # Get the magnitude range and mag limits
        #if exp_time < 60:
        #    mag_min = min(mag_range)
        #elif exp_time < 300:
        #    mag_min = min(mag_range) + 1
        #else:
        #    mag_min = min(mag_range) + 2

        mag_max = max(mag_range)
        mag_limit = mag_limit

        sdss_query = f"""
        SELECT top {num_sources} p.objID, p.ra, p.dec, p.{filter}, p.clean
        FROM PhotoObj p
        JOIN dbo.fGetObjFromRectEq({ra_min+0.03}, {dec_min+0.03}, {ra_max-0.03}, {dec_max-0.03}) r ON p.objID = r.objID
        WHERE p.clean = 1 AND p.{filter} > {mag_max - 2} AND p.{filter} < {mag_max}
        ORDER BY p.{filter} ASC
        """
        print(sdss_query)

        job = SDSS.query_sql(sdss_query)
        if self.PS:
            job = None
        # if the job is not empty but is less than 100 sources, try to get more sources iterate over this loop
        i = 0
        while job is not None and len(job) < 100 and mag_max < mag_limit:
            i  = i + 1
            mag_max = mag_max + 1
            sdss_query = f"""
            SELECT top {num_sources} p.objID, p.ra, p.dec, p.{filter}, p.clean
            FROM PhotoObj p
            JOIN dbo.fGetObjFromRectEq({ra_min+0.03}, {dec_min+0.03}, {ra_max-0.03}, {dec_max-0.03}) r ON p.objID = r.objID
            WHERE p.clean = 1 AND p.{filter} > {mag_max - (3 + i)} AND p.{filter} < {mag_max}
            ORDER BY p.{filter} ASC
            """
            print(sdss_query)
            job = SDSS.query_sql(sdss_query)

        if job is None:
            PS = True
            # If the query fails, try querying Vizer Pan-STARRS instead
            print("SDSS query failed, trying Pan-STARRS")
            job = self.get_pan_starrs_catalog(wcs, filter, mag_range, num_sources=500)
            #print(job)

        return job, PS

    def get_pan_starrs_catalog(self, wcs, filter, mag_range=(12, 16), num_sources=300, mag_limit=20):
        # Get the bounds of the image in sky coordinates
        ra_min, ra_max, dec_min, dec_max = self.get_image_bounds(wcs)
        
        # Get the exposure time
        #self.header = fits.getheader(self.image_path)
        #exp_time = self.header['EXPTIME']

        # Get the magnitude range and mag limits
        #if exp_time < 30:
        #    mag_min = min(mag_range)
        #elif exp_time < 120:
        #    mag_min = min(mag_range) + 1
        #else:
        #    mag_min = min(mag_range) + 2

        mag_max = max(mag_range)
        mag_limit = mag_limit

        # Define the region to query with a 0.05 degree buffer
        width = (ra_max - ra_min) - 0.05
        height = (dec_max - dec_min) - 0.05
        ra_center = (ra_max + ra_min) / 2
        dec_center = (dec_max + dec_min) / 2
        region = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg), frame='icrs')
    
        # Query Pan-STARRS catalog
        mag_column = f"{filter}mag"
        v = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000', mag_column, 'gmag', 'rmag'], 
                   column_filters={mag_column: f">{mag_max - 2} & <{mag_max}"}, 
                   row_limit=num_sources*100)
        result = v.query_region(region, width=width*u.deg, height=height*u.deg, catalog='II/349/ps1')

        sorted_result = result[0].to_pandas().sort_values(by=[mag_column], ascending=True)
        top_sources = sorted_result[:num_sources]
        #turn back into astropy table
        top_result = Table.from_pandas(top_sources)

        if len(result) == 0:
            print("No sources found in Pan-STARRS catalog.")
            return None
        i = 0
        # if top_result is not empty but is less than 100 sources, try to get more sources and iterate over this loop
        while top_result is not None and len(top_result) < 100 and mag_max < mag_limit:
            mag_max = mag_max + 1
            i = i + 1
            v = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000', mag_column, 'gmag', 'rmag'], 
                       column_filters={mag_column: f">{mag_max - (3 + i)} & <{mag_max}"}, 
                       row_limit=num_sources*100)
            result = v.query_region(region, width=width*u.deg, height=height*u.deg, catalog='II/349/ps1')
            sorted_result = result[0].to_pandas().sort_values(by=[mag_column], ascending=True)
            top_sources = sorted_result[:num_sources]
            #turn back into astropy table
            top_result = Table.from_pandas(top_sources)

        return top_result


class ImageProcessor:
    def __init__(self, image, PS=False):
        self.image_path = image
        self.data, self.header = fits.getdata(image, header=True)
        self.wcs = WCS(self.header)  # Ensure consistent WCS assignment
        #if no WCS in header then print error
        if not self.wcs or not self.wcs.has_celestial:
            print("No valid WCS in header")
            #stop the program
            exit()
        self.sdss_catalog = SDSS_catalog(PS=PS)

    def find_sources(self):
        mean, median, std = sigma_clipped_stats(self.data, sigma=5.0)

        try:
            fwhmval = float(str(self.header['FWHM']))
        except:
            try:
                fwhmval = (float(str(self.header['FWHMH'])) + float(str(self.header['FWHMV']))) / 2
                print('No FWHM found, using horizontal/vertical average')
            except:
                print('No FWHM data in header, defaulting to 8px')
                fwhmval = 8

        daofind = DAOStarFinder(fwhm=fwhmval, threshold=15 * std, exclude_border=True)
        sources = daofind(self.data)

        for col in sources.colnames:
            sources[col].info.format = '%.8g'

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        min_distance = 6 * fwhmval

        keep_sources = []
        for x, y in positions:
            if all(np.sqrt((x - x0)**2 + (y - y0)**2) > min_distance for x0, y0 in keep_sources):
                keep_sources.append((x, y))
        
        # Convert to numpy array for consistency
        positions = np.array(keep_sources)

        return positions[1:-1]
    
        
    def compute_fwhm(self, positions, size=30):
        # Use SExtractor to compute the FWHM of the sources
        from astropy.nddata import Cutout2D
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground
        from photutils.psf import extract_stars
        from photutils.psf import DAOGroup
        from photutils.psf import IntegratedGaussianPRF
        from photutils.psf import IterativelySubtractedPSFPhotometry

        bkg_estimator = MedianBackground()
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(self.data, (100, 100), filter_size=(5, 5),
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        
        data = self.data.astype(float)
        data -= bkg.background

        # Make a segmentation image
        from photutils import detect_sources
        from photutils import detect_threshold
        from photutils import source_properties

        threshold = detect_threshold(data, nsigma=3.)
        npixels = 5  # minimum number of connected pixels
        segm = detect_sources(data, threshold, npixels)
        props = source_properties(data, segm)
        fwhm = props.fwhm

        median_fwhm = np.median(fwhm)

        return fwhm, median_fwhm

    
    def compute_fwhm_pixel_method(self, positions, size=30):
        from astropy.nddata import Cutout2D
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

        fwhm_list = []

        bkg_estimator = MedianBackground()
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(self.data, (100, 100), filter_size=(5, 5),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        data = self.data.astype(float)
        data -= bkg.background

        for x, y in positions:
            x, y = int(x), int(y)
            if (x - size < 0 or x + size >= data.shape[1] or
                    y - size < 0 or y + size >= data.shape[0]):
                continue

            cutout = data[y - size:y + size + 1, x - size:x + size + 1]
            peak = cutout.max()
            half_max = 0.5 * peak
            y_indices, x_indices = np.where(cutout >= half_max)

            if len(x_indices) > 0 and len(y_indices) > 0:
                fwhm_x = max(x_indices) - min(x_indices)
                fwhm_y = max(y_indices) - min(y_indices)
                fwhm = (fwhm_x + fwhm_y) / 2
                fwhm_list.append(fwhm)

        fwhm_median = np.median(fwhm_list)
        # Make an entry to the header for the FWHM
        self.header['FWHM'] = fwhm_median
        #write FWHM to the header
        with fits.open(self.image_path, mode="update") as hdul:
            hdul[0].header['FWHM'] = fwhm_median

        return fwhm_list, fwhm_median

    def sdss_check(self, filter=None):
        if filter is None:
            filter = self.header.get('FILTER').lower()
        catalog, PS = self.sdss_catalog.get_catalog(self.wcs, filter=filter)
        sdss_sources = []
        color = []

        for row in catalog:
            try:
                ra, dec, mag, clean = row['ra'], row['dec'], row[filter], row['clean']
                if clean == 1:
                    sdss_sources.append((ra, dec, mag))
            except:
                mag_column = f"{filter}mag"
                ra, dec, mag, gmag, rmag = row['RAJ2000'], row['DEJ2000'], row[mag_column], row['gmag'], row['rmag']
                sdss_sources.append((ra, dec, mag))
                color.append(gmag - rmag)
        
        return sdss_sources, color, PS

    def ap_phot_mags(self, pixel_positions):

        #Calculate background value for the whole image
        #img_bkg_mean, img_bkg_median, img_bkg_std = sigma_clipped_stats(self.data, sigma=3.0)

        FWHM = float(str(self.header['FWHM']))

        radius = FWHM * 2.5
        width = 8
        buffer = 10

        apertures = CircularAperture(pixel_positions, r=radius)
        sky_annulus = CircularAnnulus(pixel_positions, r_in=radius + buffer, r_out=radius + buffer + width)

        phot_table = aperture_photometry(self.data, apertures, method='exact', wcs=self.wcs)
        
        # Initialize saturation mask for sources
        saturated_sources = []

        # Check for pixel-level saturation
        for i, _ in enumerate(pixel_positions):
            aperture_mask = apertures.to_mask(method='center')[i]  # Get mask for the aperture
            aperture_data = aperture_mask.multiply(self.data)  # Extract aperture pixels
            aperture_pixels = aperture_data[aperture_mask.data > 0]  # Only valid pixels within the mask

            # Check if any pixel exceeds 60,000
            if np.any(aperture_pixels > 60000):
                saturated_sources.append(i)

        # Convert to numpy array for easier indexing
        aperture_sum = np.array(phot_table['aperture_sum'], dtype=np.float64)

        # Set aperture sums of saturated sources to NaN
        aperture_sum[saturated_sources] = np.nan
        phot_table['aperture_sum'] = aperture_sum  # Update the photometry table 
        
        bkgmean = []

        for i in range(len(pixel_positions)):
            sky_mask = sky_annulus.to_mask(method='center')[i]
            sky_data = sky_mask.multiply(self.data)
            sky_data_1d = sky_data[sky_mask.data > 0]
            mean, _, _ = sigma_clipped_stats(sky_data_1d, sigma=3.0)
            bkgmean.append(mean)

        egain = self.header['EGAIN']
        gain = 4*egain

        bkgmean = np.array(bkgmean)
        ap_flux = gain*phot_table['aperture_sum'] - (gain*bkgmean * apertures.area)
        exp = self.header['EXPTIME']
        ap_mag = -2.5 * np.log10(ap_flux / exp)

        return ap_mag
    

    def zmag_calc(self):
        star_positions = self.find_sources()
        filter = self.header.get('FILTER').lower()
        sdss_data, color, PS = self.sdss_check(filter=filter)

        try:
            sdss_positions = np.array([[ra, dec] for ra, dec, mag in sdss_data])
            sdss_mags = np.array([mag for ra, dec, mag in sdss_data])
            color = np.array([c for c in color])
            if (color.size > 0):
                if (filter == 'g'):
                    sdss_mags = sdss_mags + 0.014 + 0.162*color
                elif (filter == 'r'):
                    sdss_mags = sdss_mags - 0.001 + 0.011*color 
                elif (filter == 'i'):
                    sdss_mags = sdss_mags - 0.004 + 0.020*color
                elif (filter == 'z'):
                    sdss_mags = sdss_mags + 0.013 - 0.050*color  
        except:
            print('No SDSS sources found')
            return 'SDSS Error - no sources found'

        source_positions = self.wcs.all_world2pix(sdss_positions, 0)
        #print("Pixel Positions Before Filtering:", source_positions)
        # Filter valid pixel positions
        height, width = self.data.shape
        valid_pixels = (source_positions[:, 0] >= 0) & (source_positions[:, 0] < width) & \
            (source_positions[:, 1] >= 0) & (source_positions[:, 1] < height)
        source_positions = source_positions[valid_pixels]
        sdss_mags = sdss_mags[valid_pixels]
        sdss_catalog = np.column_stack((source_positions, sdss_mags))

        if len(source_positions) == 0:
            raise ValueError("No valid pixel positions after filtering.")
        FWHM = float(str(self.header['FWHM']))
        #Match the SDSS sources to the detected sources
        from scipy.spatial import KDTree
        tree = KDTree(star_positions)
        distances, indices = tree.query(sdss_catalog[:, :2])
        sdss_catalog = sdss_catalog[distances < FWHM * 3]
        star_positions = star_positions[indices[distances < FWHM * 3]]

        sdss_positions = sdss_catalog[:, :2]
        sdss_mags = sdss_catalog[:, 2]
        
        ap_mag = self.ap_phot_mags(star_positions)
        ap_mag = np.array(ap_mag)

        zmaglist = sdss_mags - ap_mag
        ap_maglist = ap_mag
        sdss_maglist = sdss_mags

        #filter out sources with NaN values in either list
        mask = ~np.isnan(zmaglist) & ~np.isnan(ap_maglist) & ~np.isnan(sdss_maglist)
        zmaglist = zmaglist[mask]
        ap_maglist = ap_maglist[mask]
        sdss_maglist = sdss_maglist[mask]

        # Convert lists to NumPy arrays
        zmaglist = np.array(zmaglist)
        ap_maglist = np.array(ap_maglist)
        sdss_maglist = np.array(sdss_maglist)
        # exclude outliers from ap_maglist and sdss_maglist
        median_zmag = np.median(zmaglist)
        err = np.std(zmaglist)
        err_mask = (zmaglist < median_zmag + 2 * err) & (zmaglist > median_zmag - 2 * err)
        zmaglist = zmaglist[err_mask]
        ap_maglist = ap_maglist[err_mask]
        sdss_maglist = sdss_maglist[err_mask]
        fit = np.polyfit(ap_maglist, sdss_maglist, 1)
        p = np.poly1d(fit)
        # calc the error on the fit
        fiterr = np.sqrt(np.sum((p(ap_maglist) - sdss_maglist)**2) / (len(sdss_maglist) - 2))

        for _ in range(1):
            zmaglist_new = []
            ap_maglist_new = []
            sdss_maglist_new = []

            for i in range(len(ap_maglist)):
                if not (np.isnan(ap_maglist[i]) or np.isnan(sdss_maglist[i])):
                    if abs(sdss_maglist[i] - p(ap_maglist[i])) < 2*fiterr:
                        zmaglist_new.append(sdss_maglist[i] - ap_maglist[i])
                        sdss_maglist_new.append(sdss_maglist[i])
                        ap_maglist_new.append(ap_maglist[i])

            fit = np.polyfit(ap_maglist_new, sdss_maglist_new, 1)
            p = np.poly1d(fit)
            fiterr = np.sqrt(np.sum((p(ap_maglist_new) - sdss_maglist_new)**2) / (len(sdss_maglist_new) - 2))

        zmaglist = np.array(zmaglist_new)
        zmag_std = np.std(zmaglist_new)
        if zmag_std > 0.5:
            zmag_std = 0.5
        zmag_avg = np.mean(zmaglist_new)
        diff = np.abs(zmaglist - zmag_avg)
        zmaglist = zmaglist[(diff < 2*zmag_std)] 
        ap_maglist = np.array(ap_maglist_new)
        sdss_maglist = np.array(sdss_maglist_new)
        ap_maglist = ap_maglist[(diff < 2*zmag_std)]
        sdss_maglist = sdss_maglist[(diff < 2*zmag_std)]
        fit = np.polyfit(ap_maglist, sdss_maglist, 1)
        p = np.poly1d(fit)
        fiterr = np.sqrt(np.sum((p(ap_maglist) - sdss_maglist)**2) / (len(sdss_maglist) - 2))

        if len(zmaglist_new) < 5:
            print('Not enough sources to calculate ZMAG')
            return 'Not enough sources to calculate ZMAG'

        if args.plot:
            plt.scatter(ap_maglist, sdss_maglist)
            plt.plot(ap_maglist, p(ap_maglist))
            plt.xlabel('Instrumental Magnitude')
            plt.ylabel('SDSS Magnitude')
            plt.title('SDSS vs. Instrumental Magnitudes')
            plt.show()
        
        if args.plotZP:
            plt.scatter(sdss_maglist, zmaglist)
            plt.xlabel('SDSS Magnitude')
            plt.ylabel('ZMAG')
            plt.title('ZMAG vs. SDSS')
            plt.show()

        calc_zmag = round(sum(zmaglist) / float(len(zmaglist)), 3)
        calc_zmag_err = round(np.std(zmaglist), 3)
        #calc_zmag = round(np.median(zmaglist), 3)
        lin_fit_zmag = round(p[0], 3)
        lin_fit_zmag_err = round(fiterr, 3)

        return calc_zmag, calc_zmag_err, lin_fit_zmag, lin_fit_zmag_err, sdss_positions

    def plot_stars(self, pixel_positions):
        self.header = fits.getheader(self.image_path)
        FWHM = float(str(self.header['FWHM']))
        apertures = CircularAperture(pixel_positions, r=FWHM * 2.5)
        #sdss_data, color = self.sdss_check()
        #sdss_positions = np.array([[ra, dec] for ra, dec, mag in sdss_data])
        #sdss_positions = self.wcs.all_world2pix(sdss_positions, 0)
        #sdss_apertures = CircularAperture(sdss_positions, r=10)
        plt.figure()
        vmin = np.percentile(self.data, 5)
        vmax = np.percentile(self.data, 95)
        plt.imshow(self.data, cmap='Greys', origin='lower', vmin=vmin, vmax=vmax)
        apertures.plot(color='magenta', lw=0.5, alpha=0.5)
        #sdss_apertures.plot(color='magenta', lw=1.5, alpha=0.5)
        plt.title('Matched Sources (Magenta)')
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ZMagCalc on a FITS image.")
    parser.add_argument("path", type=str, help="Path to the FITS image files to process.")
    parser.add_argument("--plot", action="store_true", help="Display plots of instrumental vs. SDSS magnitudes.")
    parser.add_argument("--show", action="store_true", help="Display plots of detected sources.")
    parser.add_argument("--write", action="store_true", help="Write the ZPMAG to the FITS header.")
    parser.add_argument("--plotZP", action="store_true", help="Plot the SDSS magnitude vs the ZPMAG.")
    parser.add_argument("--PS1", action="store_true", default=False, help="Use Pan-STARRS instead of SDSS.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite over a previous ZPMAG file with the new ZPMAG value.")
    parser.add_argument("--writeAVG", action="store_true", help="Write the average ZPMAG to the FITS header.")
    parser.add_argument("--rewriteAVG", action="store_true", help="Rewrite over a previous ZPMAG file with the new average ZPMAG value.")
    args = parser.parse_args()

    file_dir = os.path.dirname(args.path)

    # grab .fts, .fit and .fits
    files = []
    for ext in ('fts', 'fit', 'fits'):
        files.extend(glob.glob(os.path.join(file_dir, f'*.{ext}')))

    #remove duplicates & sort
    files = sorted(set(files))
    
    print("Found files:", files)

    if args.rewrite:
        for file in files:
            im = fits.open(file)
            print(f"Calculating ZPMAG for {file}")
            processor = ImageProcessor(file, PS=args.PS1)
            try:
                fwhm_list, fwhm_median = processor.compute_fwhm_pixel_method(processor.find_sources())
                calc_zmag, zmag_err, lin_fit_zmag, fit_err, _ = processor.zmag_calc()
            except ValueError as e:
                print(f"Error calculating ZPMAG: {e}")
                continue
            if args.show:
                _, _, _, _, stars = processor.zmag_calc()
                processor.plot_stars(stars)
            with fits.open(file, mode="update") as hdul:
                hdul[0].header['ZPMAG'] = lin_fit_zmag
                hdul[0].header['Z_ERR'] = fit_err

            print(f"Average ZMAG: {calc_zmag} ± {zmag_err}")
            print(f"Linear Fit ZMAG: {lin_fit_zmag} ± {fit_err}")
    elif args.rewriteAVG:
        for file in files:
            im = fits.open(file)
            print(f"Calculating ZPMAG for {file}")
            processor = ImageProcessor(file, PS=args.PS1)
            try:
                fwhm_list, fwhm_median = processor.compute_fwhm_pixel_method(processor.find_sources())
                calc_zmag, zmag_err, lin_fit_zmag, fit_err, _ = processor.zmag_calc()
            except ValueError as e:
                print(f"Error calculating ZPMAG: {e}")
                continue
            if args.show:
                _, _, _, _, stars = processor.zmag_calc()
                processor.plot_stars(stars)
            with fits.open(file, mode="update") as hdul:
                hdul[0].header['ZPMAG'] = calc_zmag
                hdul[0].header['Z_ERR'] = zmag_err

            print(f"Average ZMAG: {calc_zmag} ± {zmag_err}")
            print(f"Linear Fit ZMAG: {lin_fit_zmag} ± {fit_err}")
    else:
    # check if the file has ZPMAG in the header
        for file in files:
            header = fits.getheader(file)
            if 'ZPMAG' in header:
                print(f"File {file} already has ZPMAG in the header.")
                continue
            else:
                im = fits.open(file)
                print(f"Calculating ZPMAG for {file}")
                processor = ImageProcessor(file, PS=args.PS1)

                fwhm_list, fwhm_median = processor.compute_fwhm_pixel_method(processor.find_sources())
                calc_zmag, zmag_err, lin_fit_zmag, fit_err, _ = processor.zmag_calc()
                if args.show:
                    _, _, _, _, stars = processor.zmag_calc()
                    processor.plot_stars(stars)
                if args.write:
                    with fits.open(file, mode="update") as hdul:
                        hdul[0].header['ZPMAG'] = lin_fit_zmag
                        hdul[0].header['Z_ERR'] = fit_err
                if args.writeAVG:
                    with fits.open(file, mode="update") as hdul:
                        hdul[0].header['ZPMAG'] = calc_zmag
                        hdul[0].header['Z_ERR'] = zmag_err

                print(f"Average ZMAG: {calc_zmag} ± {zmag_err}")
                print(f"Linear Fit ZMAG: {lin_fit_zmag} ± {fit_err}")
    
