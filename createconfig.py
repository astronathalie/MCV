from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.stats import SigmaClip
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
import astropy.units as u
import numpy as np
import configparser


def find_target(TARGET):
    Simbad.add_votable_fields('pmra', 'pmdec')
    result = Simbad.query_object(TARGET)
    src = SkyCoord(
        ra=result['ra'][0], dec=result['dec'][0],
        unit=(u.deg, u.deg),
        pm_ra_cosdec=result['pmra'][0]*u.mas/u.yr,
        pm_dec=result['pmdec'][0]*u.mas/u.yr,
        obstime=Time('J2000.0'), frame='icrs'
    )
    current = src.apply_space_motion(new_obstime=Time('2026-04-11'))
    target_ra = current.ra.deg
    target_dec = current.dec.deg
    print(f"Target: {TARGET} ({target_ra:.5f}, {target_dec:.5f})")
    return src, target_ra, target_dec

def find_comparison_stars(target_ra, target_dec, src, REF_IMAGE_NAME, N_STARS, EDGE_BUFFER, MIN_SEP_TARGET, AP_R, SAT_LIMIT):
    hdulist = fits.open(REF_IMAGE_NAME)
    try:
        plotdata = hdulist[1].data
        plothdr = hdulist[1].header
    except Exception:
        plotdata = hdulist[0].data
        plothdr = hdulist[0].header
    hdulist.close()
    plotwcs = WCS(plothdr)
    ny, nx = plotdata.shape

    # ---- QUERY VSX FOR KNOWN VARIABLES ----
    Vizier.ROW_LIMIT = -1
    vsx_result = Vizier.query_region(src, radius=20*u.arcmin, catalog="B/vsx")
    if vsx_result:
        vsx_coords = SkyCoord(ra=vsx_result[0]['RAJ2000'], dec=vsx_result[0]['DEJ2000'], unit='deg')
        print(f"VSX: {len(vsx_coords)} known variables in field")
    else:
        vsx_coords = None
        print("No known variables in field")
    # ---- Overide radius of search ---
    # Change RADIUS to change the search radius
    # for the comparison stars

    RADIUS = 15*u.arcmin

    # ---- QUERY VIZIER ----
    Gaia.ROW_LIMIT = 1000
    field_center = SkyCoord(target_ra, target_dec, unit='deg')
    # gaia_query = Gaia.cone_search_async(field_center, radius=RADIUS)
    # gaia = gaia_query.get_results()
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(field_center, radius=15*u.arcmin, catalog="I/355/gaiadr3")
    gaia = result[0]
    print(f"Gaia: {len(gaia)} sources in field")

    gaia_sc = SkyCoord(ra=gaia['RA_ICRS'], dec=gaia['DE_ICRS'], unit='deg')
    gaia_x, gaia_y = plotwcs.wcs_world2pix(gaia_sc.ra.deg, gaia_sc.dec.deg, 0)
    target_x, target_y = plotwcs.wcs_world2pix(target_ra, target_dec, 0)

    target_match_idx = np.argmin(gaia_sc.separation(field_center))
    target_gmag = float(gaia['Gmag'][target_match_idx])
    print(f"Target G = {target_gmag:.2f}")

    MAG_RANGE = (target_gmag - 2.0, target_gmag + 2.0)

    # ---- FILTER AND RANK ----
    sigclip = SigmaClip(sigma=3., maxiters=10)
    candidates = []

    for i in range(len(gaia)):
        x, y = float(gaia_x[i]), float(gaia_y[i])

        if x < EDGE_BUFFER or x > nx - EDGE_BUFFER or y < EDGE_BUFFER or y > ny - EDGE_BUFFER:
            continue
        if np.hypot(x - float(target_x), y - float(target_y)) < MIN_SEP_TARGET:
            continue
        if vsx_coords is not None:
            sep = gaia_sc[i].separation(vsx_coords)
            if sep.min() < 5 * u.arcsec:
                continue

        ix, iy = int(round(x)), int(round(y))
        stamp = plotdata[max(0,iy-AP_R):iy+AP_R, max(0,ix-AP_R):ix+AP_R]
        if stamp.size == 0 or np.any(stamp > SAT_LIMIT):
            continue

        ap = CircularAperture((x, y), r=AP_R)
        an = CircularAnnulus((x, y), r_in=AP_R+10, r_out=AP_R+15)
        bkg_stats = ApertureStats(plotdata, an, sigma_clip=sigclip)
        ap_stats = ApertureStats(plotdata, ap)

        source_sum = ap_stats.sum - bkg_stats.median * ap.area
        noise = np.sqrt(ap_stats.sum + ap.area * bkg_stats.std**2)

        gmag = float(gaia['Gmag'][i])
        if not (MAG_RANGE[0] <= gmag <= MAG_RANGE[1]):
            continue

        if source_sum <= 0 or noise <= 0:
            continue

        snr = source_sum / noise
        candidates.append({'x': x, 'y': y, 'snr': snr, 'gmag': float(gaia['Gmag'][i])})

    candidates = sorted(candidates, key=lambda c: c['snr'], reverse=True)
    comp_stars = candidates[:N_STARS]

    # ---- OVERRIDE: uncomment below to manually exclude or pick stars ----
    # The candidates list is ranked by SNR. Indices shown in printout above.
    #
    # Option A: exclude specific stars by index, auto-fill from next best
    # EXCLUDE = [4]
    # comp_stars = [c for j, c in enumerate(candidates[:N_STARS + len(EXCLUDE)]) if j not in EXCLUDE][:N_STARS]
    #
    # Option B: hand-pick which candidates to use
    # USE = [0, 1, 3, 5, 7]
    # comp_stars = [candidates[i] for i in USE]

    print(f"\nTop 15 candidates (first {N_STARS} auto-selected with *):")
    for j, s in enumerate(candidates[:15]):
        marker = " *" if j < N_STARS else ""
        print(f"  [{j}] ({s['x']:.2f}, {s['y']:.2f})  G={s['gmag']:.2f}  SNR={s['snr']:.1f}{marker}")

    return comp_stars

def write_config(comp_stars, TARGET, PERIOD, IS_PHASE, PHASE_ZERO, REF_IMAGE_NAME, output):
    TARGET_NAME = TARGET.replace(' ', '_')
    star_coords = [f"{s['x']:.4f}" + "," + f"{s['y']:.4f}" for s in comp_stars]
    # Flatten to single comma-separated list: x1,y1,x2,y2,...
    STARS = ','.join(star_coords)

    config = configparser.ConfigParser()
    config['Target'] = {'Target': TARGET_NAME, 'Period': str(PERIOD), 'Is Phase': str(IS_PHASE), 'Phase zero': str(PHASE_ZERO)}
    config['Reference'] = {'Ref image': REF_IMAGE_NAME, 'Ref stars': STARS}

    config_path = output + TARGET_NAME + '_config.ini'
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    print(f"\nConfig saved to {config_path}")

    return config_path
