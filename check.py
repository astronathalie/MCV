
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

def get_target_coordinates(TARGET_NAME, REFERENCE_IMAGE, non_sid=False, non_sid_id=None, non_sid_loc=None):
    """
    Get the RA and Dec coordinates of the target.

    Parameters:
    TARGET_NAME (str): Name of the target object.
    non_sid (bool): Flag indicating if the target is a non-sidereal object.
    non_sid_id (str): ID of the non-sidereal object (if applicable).
    non_sid_loc (str): Location code for the non-sidereal object (if applicable).
    REFERENCE_IMAGE (str): Path to the reference image file.

    Returns:
    tuple: RA and Dec coordinates in degrees.
    """
    # get the ra, dec coordinates of the target
    if not non_sid:
        Simbad.add_votable_fields('pmra', 'pmdec')
        result = Simbad.query_object(TARGET_NAME)

        src = SkyCoord(
            ra=result['ra'][0],
            dec=result['dec'][0],
            unit=(u.deg, u.deg),
            pm_ra_cosdec=result['pmra'][0]*u.mas/u.yr,
            pm_dec=result['pmdec'][0]*u.mas/u.yr,
            obstime=Time('J2000.0'),
            frame='icrs'
        )

        current = src.apply_space_motion(new_obstime=Time('2026-04-10'))
        target_ra = current.ra.deg
        target_dec = current.dec.deg
        print(target_ra)
    else:
        try:
            hdr = fits.getheader(REFERENCE_IMAGE, ext=0)
            img_jd=hdr['JD']
            non_sid_obj=Horizons(id=non_sid_id, location=non_sid_loc, epochs=img_jd)
            eph=non_sid_obj.ephemerides()
            target_ra=eph['RA'][0]
            target_dec=eph['DEC'][0]
            src = coord.SkyCoord(target_ra, target_dec, unit="deg")
        except Exception:
            hdr = fits.getheader(REFERENCE_IMAGE, ext=1)
            img_jd=hdr['JD']
            non_sid_obj=Horizons(id=non_sid_id, location=non_sid_loc, epochs=img_jd)
            eph=non_sid_obj.ephemerides()
            target_ra=eph['RA'][0]
            target_dec=eph['DEC'][0]
            src = coord.SkyCoord(target_ra, target_dec, unit="deg")

    print(f"Target {TARGET_NAME} ({target_ra} deg, {target_dec} deg)\n")

    # Load the reference image:
    hdulist = fits.open(REFERENCE_IMAGE)
    try:
        plotdata = hdulist[1].data
        plothdr = hdulist[1].header
        hdulist.close()
        plotwcs = WCS(plothdr)
    except Exception:
        plotdata = hdulist[0].data
        plothdr = hdulist[0].header
        hdulist.close()
        plotwcs = WCS(plothdr)

    # ---- VARIABLE STAR SEARCH ----
    Vizier.ROW_LIMIT = -1  # No row limit
    catalog = "B/vsx"  # AAVSO VSX catalog
    radius = 20 * u.arcmin

    result = Vizier.query_region(src, radius=radius, catalog=catalog)

    if result:
        vsx_table = result[0]
        print(f"Found {len(vsx_table)} variable stars within {radius.to(u.arcmin)} of {TARGET_NAME}")

        vars = []
        for star in vsx_table:
            var_coord = SkyCoord(ra=star['RAJ2000'], dec=star['DEJ2000'], unit=(u.deg, u.deg))
            x_var, y_var = plotwcs.all_world2pix(var_coord.ra.deg, var_coord.dec.deg, 0)
            #ax.add_artist(plt.Circle((x_var, y_var), 15, color='magenta', fill=False, lw=1.5))
            vars.append((x_var, y_var))

    else:
        print("No variable stars found in region.")

    return target_ra, target_dec, src, vars, vsx_table


def stars_to_radec(STAR_COORDS, REFERENCE_IMAGE):
    # Load an image:
    hdulist = fits.open(REFERENCE_IMAGE)
    try:
        plotdata = hdulist[1].data
        plothdr = hdulist[1].header
        hdulist.close()
        plotwcs = WCS(plothdr)
    except Exception:
        plotdata = hdulist[0].data
        plothdr = hdulist[0].header
        hdulist.close()
        plotwcs = WCS(plothdr)
        
    checkstars_pix = STAR_COORDS
    print("RA, DEC coordinates of check stars:")
    checkstars_radec = [plotwcs.all_pix2world(*star, 0) for star in checkstars_pix]
    print(',\n'.join([f"({x}, {y})" for (x,y) in checkstars_radec]))

    return checkstars_radec

def plot_reference_image(TARGET_NAME, target_ra, target_dec, REFERENCE_IMAGE, checkstars_radec, vsx_table, show_ref_stars=True, show_var_stars=True):
    print('Reference stars in green. Variable stars in magenta. Target in red.')

    # Load an image:
    hdulist = fits.open(REFERENCE_IMAGE)
    #hdulist = fits.open(core + "/fits/rlmt/2025-2026-observing-season/V667_Pup/2026-01-28/mns_V667_Pup_g_180s_2026-01-28T07-16-13.fts.fz")
    try:
        plotdata = hdulist[1].data
        plothdr = hdulist[1].header
        hdulist.close()
        plotwcs = WCS(plothdr)
    except Exception:
        plotdata = hdulist[0].data
        plothdr = hdulist[0].header
        hdulist.close()
        plotwcs = WCS(plothdr)

    # ---- INITIAL PLOT ----
    fig, ax = plt.subplots(figsize=(12, 9))
    norm_zscale = ImageNormalize(plotdata, interval=ZScaleInterval())
    im = ax.imshow(plotdata, origin='lower', cmap='gray', norm=norm_zscale)

    # plot the target star
    x, y = plotwcs.all_world2pix(target_ra, target_dec, 0)
    ax.add_artist(plt.Circle((x, y), 25, color='red', fill=False))

    # plot the check stars
    if show_ref_stars:
        for n, (ra, dec) in enumerate(checkstars_radec):
            x, y = plotwcs.all_world2pix(ra, dec, 0)
            ax.add_artist(plt.Circle((x,y), 25, color='lime', fill=False))
            ax.annotate(str(n), xy=(x, y), xytext=(10, 10), textcoords='offset pixels')

    if show_var_stars:
        for star in vsx_table:
                var_coord = SkyCoord(ra=star['RAJ2000'], dec=star['DEJ2000'], unit=(u.deg, u.deg))
                x_var, y_var = plotwcs.all_world2pix(var_coord.ra.deg, var_coord.dec.deg, 0)
                ax.add_artist(plt.Circle((x_var, y_var), 15, color='magenta', fill=False, lw=1.5))

    # Create colorbar axis next to image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 5% width, small gap
    fig.colorbar(im, cax=cax)
    ax.set_title(TARGET_NAME + ' Check')
    plt.show()


