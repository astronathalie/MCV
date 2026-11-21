import os
from astropy import wcs
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

SDSS_BANDS = ('u', 'g', 'r', 'i', 'z')

# File types to pick up when given a folder (.fz = fpack-compressed)
FITS_EXTENSIONS = ('.fits', '.fit', '.fts', '.fits.fz', '.fit.fz', '.fts.fz')


def image_hdu_index(hdul):
    """Index of the HDU holding the image: 0 for plain FITS, 1 for .fz
    files (their primary HDU is empty and the image lives in a compressed
    extension, along with the real header: WCS, FILTER, EXPTIME...)."""
    for i, hdu in enumerate(hdul):
        if hdu.is_image and hdu.header.get('NAXIS', 0) >= 2:
            return i
    raise ValueError("No image data found in file")


def read_image_header(path):
    """The header that goes with the image (works for .fits and .fts.fz)."""
    with fits.open(path) as hdul:
        return hdul[image_hdu_index(hdul)].header.copy()


def update_image_header(path, set_cards=None, remove=()):
    """Change keywords in the image's own header, in place (works for .fz too).

    For .fz files this edits the compressed table's header directly, so the
    pixels are never decompressed and recompressed. (Going through astropy's
    normal compressed-image route re-quantizes float images on every save,
    which slightly changes the pixel values each time.)"""
    with fits.open(path, mode="update", disable_image_compression=True) as hdul:
        idx = next((i for i, h in enumerate(hdul) if h.header.get('ZIMAGE')), None)
        if idx is None:
            idx = image_hdu_index(hdul)
        hdr = hdul[idx].header
        for key, value in (set_cards or {}).items():
            hdr[key] = value
        for key in remove:
            hdr.pop(key, None)

# The zero point is always computed with the slope held at 1 (sigma-clipped
# mean of catalogue - instrumental). A free-slope fit is done only as a quality
# check: an error is printed when its slope is more than SLOPE_TOL from 1.
SLOPE_TOL = 0.10
MIN_STARS = 5
CLIP_SIGMA = 3.0


def slope1_zeropoint(inst_mag, cat_mag, sigma=CLIP_SIGMA):
    """Zero point with the slope of catalogue vs instrumental magnitude held at 1.

    That is just the mean of (catalogue - instrumental); here a sigma-clipped
    mean (clipping about the median, using the robust MAD scatter).

    Returns (zmag, zmag_err, zmag_std, slope, n, keep):
      zmag      sigma-clipped mean zero point
      zmag_err  uncertainty on zmag = zmag_std / sqrt(n)
      zmag_std  star-to-star scatter of the stars that survived clipping
      slope     slope of a free line fit to the same stars (quality check only)
      n         number of stars used
      keep      boolean mask of the stars used"""
    from astropy.stats import sigma_clip
    inst_mag = np.asarray(inst_mag, dtype=float)
    cat_mag = np.asarray(cat_mag, dtype=float)
    diff = cat_mag - inst_mag
    idx = np.flatnonzero(np.isfinite(diff))
    if idx.size < MIN_STARS:
        raise ValueError(f'Not enough matched stars to calculate ZMAG ({idx.size}, need {MIN_STARS})')
    clipped = sigma_clip(diff[idx], sigma=sigma, maxiters=10,
                         cenfunc='median', stdfunc='mad_std')
    keep = np.zeros(diff.shape, dtype=bool)
    keep[idx[~np.ma.getmaskarray(clipped)]] = True
    n = int(keep.sum())
    if n < MIN_STARS:
        raise ValueError(f'Not enough matched stars to calculate ZMAG ({n}, need {MIN_STARS})')
    zmag = float(np.mean(diff[keep]))
    zmag_std = float(np.std(diff[keep], ddof=1))
    zmag_err = zmag_std / np.sqrt(n)
    if np.ptp(inst_mag[keep]) > 0:
        slope = float(np.polyfit(inst_mag[keep], cat_mag[keep], 1)[0])
    else:
        slope = float('nan')
    return zmag, zmag_err, zmag_std, slope, n, keep

def band_from_header(header):
    """Turn the FITS FILTER keyword into an SDSS band letter (u, g, r, i, z).
    Accepts forms like 'g', 'G', "g'", 'gp', 'sdss_g', 'SDSS g'.
    Returns None for filters with no catalogue match (e.g. Ha, lum, grisms)."""
    raw = str(header.get('FILTER', '')).strip().lower()
    for junk in ('sdss', "'", '_', '-', ' '):
        raw = raw.replace(junk, '')
    if len(raw) == 2 and raw.endswith('p'):
        raw = raw[0]
    return raw if raw in SDSS_BANDS else None


class SDSS_catalog:
    def __init__(self, PS = False):
        self.PS = PS
        pass

    def get_image_bounds(self, wcs):
        naxis1 = wcs.array_shape[1]
        naxis2 = wcs.array_shape[0]
        corners = [(0, 0), (0, naxis2), (naxis1, 0), (naxis1, naxis2)]
        ra, dec = wcs.all_pix2world(corners, 0).T
        
        wrapped = (ra.max() - ra.min()) > 180
        if wrapped:
            ra = np.where(ra < 180, ra + 360, ra)
        
        ra_min, ra_max = float(ra.min()), float(ra.max())
        dec_min, dec_max = float(dec.min()), float(dec.max())
        
        return ra_min, ra_max, dec_min, dec_max, wrapped

    def get_catalog(self, wcs, filter, mag_range=(12, 16), num_sources=200, mag_limit=20):
        # Get the bounds of the image in sky coordinates
        PS = self.PS
        ra_min, ra_max, dec_min, dec_max, wrapped = self.get_image_bounds(wcs)

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
        # Detect HTML error response masquerading as a table
        def _is_valid_sdss(t):
            if t is None:
                return False
            expected = {'objID', 'ra', 'dec', 'clean'}
            return expected.issubset(set(t.colnames))

        if not _is_valid_sdss(job):
            print("SDSS returned invalid response (likely HTML error page)")
            job = None
        
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
            # Detect HTML error response masquerading as a table
            def _is_valid_sdss(t):
                if t is None:
                    return False
                expected = {'objID', 'ra', 'dec', 'clean'}
                return expected.issubset(set(t.colnames))

            if not _is_valid_sdss(job):
                print("SDSS returned invalid response (likely HTML error page)")
                job = None
            if mag_max >= mag_limit and num_sources < 1:
                print("No sources found in SDSS catalog, and mag limit reached.")
                job = None

        if job is None:
            PS = True
            # If the query fails, try querying Vizer Pan-STARRS instead
            print("SDSS query failed, trying Pan-STARRS")
            job = self.get_pan_starrs_catalog(wcs, filter, mag_range, num_sources=500)
            #print(job)

        return job, PS

    def get_pan_starrs_catalog(self, wcs, filter, mag_range=(12, 16), num_sources=300, mag_limit=20):
        # Get the bounds of the image in sky coordinates
        ra_min, ra_max, dec_min, dec_max, wrapped = self.get_image_bounds(wcs)
        
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

        if len(result) == 0 or len(result[0]) == 0:
            print("No sources found in Pan-STARRS catalog.")
            return None

        sorted_result = result[0].to_pandas().sort_values(by=[mag_column], ascending=True)
        top_sources = sorted_result[:num_sources]
        #turn back into astropy table
        top_result = Table.from_pandas(top_sources)
        i = 0
        # if top_result is not empty but is less than 100 sources, try to get more sources and iterate over this loop
        while top_result is not None and len(top_result) < 100 and mag_max < mag_limit:
            mag_max = mag_max + 1
            i = i + 1
            v = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000', mag_column, 'gmag', 'rmag'], 
                       column_filters={mag_column: f">{mag_max - (3 + i)} & <{mag_max}"}, 
                       row_limit=num_sources*100)
            result = v.query_region(region, width=width*u.deg, height=height*u.deg, catalog='II/349/ps1')
            if len(result) == 0 or len(result[0]) == 0:
                break   # keep the last non-empty list
            sorted_result = result[0].to_pandas().sort_values(by=[mag_column], ascending=True)
            top_sources = sorted_result[:num_sources]
            #turn back into astropy table
            top_result = Table.from_pandas(top_sources)

        return top_result


class ImageProcessor:
    def __init__(self, image, PS=False):
        self.image_path = image
        with fits.open(image) as hdul:
            ext = image_hdu_index(hdul)
            self.data = hdul[ext].data.astype(np.float32)
            self.header = hdul[ext].header.copy()
        self.wcs = WCS(self.header)  # Ensure consistent WCS assignment
        #if no WCS in header then print error
        if not self.wcs or not self.wcs.has_celestial:
            raise ValueError("No valid WCS in header (image not plate solved)")
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
        if sources is None or len(sources) == 0:
            raise ValueError(f"No stars detected (search FWHM {fwhmval} px)")

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

        return positions
    
        
    def compute_fwhm_pixel_method(self, positions, size=30, write_header=False):
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

        from scipy import ndimage

        for x, y in positions:
            x, y = int(round(x)), int(round(y))
            if (x - size < 0 or x + size >= data.shape[1] or
                    y - size < 0 or y + size >= data.shape[0]):
                continue

            cutout = data[y - size:y + size + 1, x - size:x + size + 1]
            # Peak of THIS star (centre of the cutout), not a brighter neighbour
            peak = cutout[size - 1:size + 2, size - 1:size + 2].max()
            if peak <= 0:
                continue
            # Pixels above half maximum that are connected to the star's centre
            labels, _ = ndimage.label(cutout >= 0.5 * peak)
            star = labels[size, size]
            if star == 0:
                continue
            area = np.count_nonzero(labels == star)
            # A round star's half-max region is a disc of diameter FWHM
            fwhm_list.append(2.0 * np.sqrt(area / np.pi))

        if not fwhm_list:
            raise ValueError("Could not measure FWHM (no usable stars away from the edges)")
        fwhm_median = round(float(np.median(fwhm_list)), 2)
        # Keep the FWHM in memory for this run's photometry
        self.header['FWHM'] = fwhm_median
        # Only change the file itself when a write flag was given
        if write_header:
            update_image_header(self.image_path, {'FWHM': fwhm_median})

        return fwhm_list, fwhm_median

    def sdss_check(self, filter=None):
        if filter is None:
            filter = band_from_header(self.header)
        catalog, PS = self.sdss_catalog.get_catalog(self.wcs, filter=filter)
        if catalog is None or len(catalog) == 0:
            raise ValueError("No catalogue stars found in SDSS or Pan-STARRS for this field")
        sdss_sources = []
        color = []

        print(catalog.colnames)
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

        # Build each star's masks once (building them inside the loop redid
        # every star's mask for every star, which got slow in rich fields)
        ap_masks = apertures.to_mask(method='center')
        sky_masks = sky_annulus.to_mask(method='center')

        # Check for pixel-level saturation
        for i, _ in enumerate(pixel_positions):
            aperture_mask = ap_masks[i]  # Get mask for the aperture
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
            sky_mask = sky_masks[i]
            sky_data = sky_mask.multiply(self.data)
            sky_data_1d = sky_data[sky_mask.data > 0]
            mean, _, _ = sigma_clipped_stats(sky_data_1d, sigma=3.0)
            bkgmean.append(mean)

        # egain = self.header['EGAIN',0.25]
        # gain = 4*egain
        gain = 1

        bkgmean = np.array(bkgmean)
        ap_flux = gain*phot_table['aperture_sum'] - (gain*bkgmean * apertures.area)
        exp = self.header['EXPTIME']
        ap_mag = -2.5 * np.log10(ap_flux / exp)

        return ap_mag
    

    def zmag_calc(self):
        star_positions = self.find_sources()
        filter = band_from_header(self.header)
        if filter is None:
            raise ValueError(f"FILTER '{self.header.get('FILTER')}' has no SDSS/Pan-STARRS band (needs u, g, r, i or z)")
        print(f"Using catalogue band: {filter}")
        sdss_data, color, PS = self.sdss_check(filter=filter)

        try:
            sdss_positions = np.array([[ra, dec] for ra, dec, mag in sdss_data])
            sdss_mags = np.array([mag for ra, dec, mag in sdss_data])
            color = np.array([c for c in color])
            # transform from Pan-STARRS to SDSS if PS is True using Tonry et al. 2012 transformations
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
            raise ValueError('No SDSS/Pan-STARRS sources found')

        
        from astropy.wcs.wcs import NoConvergence

        try:
            source_positions = self.wcs.all_world2pix(sdss_positions, 0)
            good = np.ones(len(sdss_positions), dtype=bool)
        except NoConvergence as e:
            n = len(sdss_positions)
            good = np.ones(n, dtype=bool)
            div = e.divergent
            if div is not None:
                if div.dtype == bool:
                    diverged = np.any(div, axis=1) if div.ndim == 2 else div
                    good = ~diverged
                else:
                    good[div] = False
            print(f"sdss_positions: {n}, kept={good.sum()}")
            source_positions = e.best_solution[good]
            sdss_positions = sdss_positions[good]
            # Apply the same mask to every parallel array
            sdss_mags = sdss_mags[good]
            # and any others — sdss_ids, sdss_errs, sdss_gmag, sdss_rmag, etc.
        
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

        # Zero point with the slope held at 1: sigma-clipped mean of
        # (catalogue - instrumental). NaNs (saturated / negative flux) are dropped.
        zmag, zmag_err, zmag_std, slope, n_stars, keep = slope1_zeropoint(ap_mag, sdss_mags)
        ap_maglist = ap_mag[keep]
        sdss_maglist = sdss_mags[keep]
        zmaglist = sdss_maglist - ap_maglist
        print(f"Stars used: {n_stars} of {len(ap_mag)} matched "
              f"({CLIP_SIGMA:g}-sigma clipped)")

        # Free-slope fit: quality check only, never used for the zero point
        if not np.isfinite(slope):
            print("Warning: could not fit a slope (all stars have the same instrumental magnitude)")
        elif abs(slope - 1) > SLOPE_TOL:
            print(f"ERROR: free-fit slope {slope:.3f} is more than {SLOPE_TOL:.0%} from 1; "
                  f"the photometry is suspect (sky subtraction, saturation, mismatched stars?) "
                  f"and ZMAG {zmag:.3f} should not be trusted")
        else:
            print(f"Free-fit slope {slope:.3f} (quality check only; ZMAG uses slope = 1)")

        if args.plot:
            xs = np.array([ap_maglist.min(), ap_maglist.max()])
            plt.scatter(ap_maglist, sdss_maglist)
            plt.plot(xs, xs + zmag, label=f'slope = 1, ZMAG = {zmag:.3f}')
            if np.isfinite(slope):
                icpt = np.mean(sdss_maglist) - slope * np.mean(ap_maglist)
                plt.plot(xs, slope * xs + icpt, ls='--', label=f'free fit, slope = {slope:.3f}')
            plt.xlabel('Instrumental Magnitude')
            plt.ylabel('SDSS Magnitude')
            plt.title('SDSS vs. Instrumental Magnitudes')
            plt.legend()
            plt.show()

        if args.plotZP:
            plt.scatter(sdss_maglist, zmaglist)
            plt.axhline(zmag, label=f'ZMAG = {zmag:.3f}')
            plt.axhline(zmag + zmag_std, ls=':')
            plt.axhline(zmag - zmag_std, ls=':')
            plt.xlabel('SDSS Magnitude')
            plt.ylabel('ZMAG')
            plt.title('ZMAG vs. SDSS')
            plt.legend()
            plt.show()

        return (round(zmag, 3), round(zmag_err, 4), round(zmag_std, 3),
                slope, n_stars, sdss_positions)

    def plot_stars(self, pixel_positions):
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
    parser.add_argument("--plot", action="store_true", help="Plot instrumental vs. SDSS magnitudes with the slope = 1 line and the free fit.")
    parser.add_argument("--show", action="store_true", help="Display plots of detected sources.")
    parser.add_argument("--write", action="store_true", help="Write the ZMAG (slope held at 1, sigma-clipped mean) to the FITS header.")
    parser.add_argument("--plotZP", action="store_true", help="Plot the SDSS magnitude vs the ZMAG.")
    parser.add_argument("--PS1", action="store_true", default=False, help="Use Pan-STARRS instead of SDSS.")
    parser.add_argument("--rewrite", action="store_true", help="Recalculate and overwrite an existing ZMAG in the FITS header.")
    parser.add_argument("--writeAVG", action="store_true", help="Same as --write (kept so old commands still work).")
    parser.add_argument("--rewriteAVG", action="store_true", help="Same as --rewrite (kept so old commands still work).")
    args = parser.parse_args()
    # Files are only modified when one of these flags is given
    write_any = args.write or args.writeAVG or args.rewrite or args.rewriteAVG

    target = os.path.expanduser(args.path)
    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in os.listdir(target)
                 if f.lower().endswith(FITS_EXTENSIONS)]
    elif os.path.isfile(target):
        files = [target]
    else:
        # allow a wildcard pattern, e.g. "night/*_g_*.fts.fz"
        files = [f for f in glob.glob(target) if f.lower().endswith(FITS_EXTENSIONS)]
        if not files:
            print(f"No such file or folder: {args.path}")
            raise SystemExit(1)

    files = sorted(set(files))
    print(f"Found {len(files)} file(s)")

    rewriting = args.rewrite or args.rewriteAVG
    done, skipped = [], []

    for file in files:
        # Without --rewrite/--rewriteAVG, leave images that already have a ZMAG alone
        if not rewriting and 'ZMAG' in read_image_header(file):
            print(f"File {file} already has ZMAG in the header.")
            skipped.append((file, 'already has ZMAG'))
            continue

        print(f"Calculating ZMAG for {file}")
        try:
            processor = ImageProcessor(file, PS=args.PS1)
            processor.compute_fwhm_pixel_method(processor.find_sources(), write_header=write_any)
            zmag, zmag_err, zmag_std, slope, n_stars, stars = processor.zmag_calc()
        except Exception as e:
            print(f"Skipping {file}: {e}")
            if rewriting:
                # Don't leave a stale ZMAG behind when a rewrite fails
                update_image_header(file, remove=('ZMAG', 'ZMAGERR', 'ZMAGSTD', 'ZMAGN'))
            skipped.append((file, str(e)))
            continue

        if args.show:
            processor.plot_stars(stars)

        # Every write flag stores the same slope = 1 zero point
        if write_any:
            update_image_header(file, {
                'ZMAG': (zmag, 'Zero point, slope=1, sigma-clipped mean'),
                'ZMAGERR': (zmag_err, 'Error on ZMAG = ZMAGSTD/sqrt(ZMAGN)'),
                'ZMAGSTD': (zmag_std, 'Star-to-star scatter in ZMAG'),
                'ZMAGN': (n_stars, 'Number of stars used for ZMAG'),
            })

        print(f"ZMAG: {zmag} ± {zmag_err}   (ZMAGSTD: {zmag_std}, {n_stars} stars, slope held at 1)")
        done.append((file, zmag if write_any else None))

    print("\n===== Summary =====")
    print(f"Done: {len(done)}   Skipped: {len(skipped)}")
    for file, value in done:
        print(f"  OK    {os.path.basename(file)}" + (f"  ZMAG written: {value}" if value is not None else ""))
    for file, why in skipped:
        print(f"  SKIP  {os.path.basename(file)}  ({why})")
