#!/usr/bin/env python
"""
CalcMag_K_26.py -- per-image photometric zero points from an ensemble of
catalog comparison stars.

Revision of CalcZPMag_K.py (October 2026).

Method
------
For every image: detect stars, do aperture photometry, match to a reference
catalog (Pan-STARRS1 DR1 or APASS DR9), transform catalog magnitudes into the
band of the image, and take the zero point as the sigma-clipped mean of
(catalog mag - instrumental mag) over the ensemble.  Comparison stars are in
the same frame as the target, so first-order extinction and transparency
cancel; no separate airmass correction is applied.

Catalog -> image-band transformations
-------------------------------------
Pan-STARRS1 (default):
    B, V, R_C, I_C : Tonry et al. 2012, ApJ 750, 99, Table 6 (quadratic in
                     (g-r)_P1).  Applied directly to PS1 mags -- no SDSS step.
    g, r, i        : PS1 mags as-is (PS1 system), or converted to SDSS with
                     --sdss using the same table.
APASS DR9 (--APASS):
    B, V           : APASS Johnson B, V directly.
    R_C, I_C       : Lupton (2005) from APASS r', i' (SDSS-like):
                     R = r - 0.2936 (r-i) - 0.1439
                     I = r - 1.2444 (r-i) - 0.3820
    g, r, i        : APASS g', r', i' directly.
Unfiltered ("CV", "Clear", "L", ...) frames are calibrated to V by default
(the AAVSO "CV" convention); change with --cvband.

Fixes relative to CalcZPMag_K.py
--------------------------------
 1. R and I with PS1: old code computed B - (V-R) instead of V - (V-R)
    (catalog R too faint by ~B-V, ~0.5-1 mag).  Now Tonry et al. 2012.
 2. R and I with APASS: '==' instead of '=' meant no transformation was
    applied (R was raw r').  Fixed.
 3. PS1 -> SDSS conversion never ran (case mismatch in filter check).  Now
    explicit and optional (--sdss) for g, r, i; not needed for BVRI.
 4. The default (no catalog flag) path crashed (undefined variable), so the
    SDSS query never ran.  PS1 is now the default; SDSS code removed.
 5. --write stored the intercept of a free-slope linear fit, which is not a
    zero point unless the slope is exactly 1.  ZPMAG is now always the
    sigma-clipped mean offset.  The fitted slope is kept only as a
    diagnostic (Z_SLOPE).
 6. CV frames were silently calibrated to PS1 g.  Now V by default.
 7. Catalog queried by a cone that covers the whole field (handles RA = 0
    wrap); stars too close to the edge for the sky annulus are dropped;
    blended matches (two catalog stars on one detection) are dropped;
    comparison stars restricted to a color range where the transformations
    are valid.
 8. Header records how the zero point was made (catalog, band,
    transformation, N stars, scatter, color-trend diagnostic, code version).
 9. Removed: duplicate catalog query, arbitrary dropping of the first and
    last detected sources, unused/broken FWHM routine, reliance on a global
    'args'.  FITS files are only modified when --write is given.

Usage
-----
    python CalcMag_K_26.py /path/to/images/ --write
    python CalcMag_K_26.py /path/to/images/*.fits --APASS --write --plotColor
"""

import os
import glob
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import sigma_clipped_stats, sigma_clip
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

VERSION = 'CalcMag_K_26'

# Detector constants (same values as CalcZPMag_K.py so instrumental mags are
# on the same scale as before).  Gain is a multiplicative constant and is
# absorbed into the zero point.
GAIN = 0.27          # e-/ADU
SATURATION = 60000   # ADU; apertures containing a pixel above this are dropped

# Aperture geometry (in units of FWHM for the radius, pixels for the annulus)
AP_RADIUS_FWHM = 2.5
ANNULUS_GAP = 10
ANNULUS_WIDTH = 8


# ---------------------------------------------------------------------------
# Photometric transformations
# ---------------------------------------------------------------------------

def _quad(x, coeffs):
    a0, a1, a2 = coeffs
    return a0 + a1 * x + a2 * x ** 2


# Tonry et al. 2012, ApJ 750, 99, Table 6, quadratic fits, x = (g - r)_P1.
# Each entry: target band -> (PS1 base band, (A0, A1, A2)) for
# (target - base_P1) = A0 + A1 x + A2 x^2
PS1_TO_JC = {
    'B': ('g', (0.212, 0.556, 0.034)),     # B   - g_P1   (rms 0.032)
    'V': ('r', (0.005, 0.462, 0.013)),     # V   - r_P1   (rms 0.012)
    'R': ('r', (-0.137, -0.108, -0.029)),  # R_C - r_P1   (rms 0.015)
    'I': ('i', (-0.366, -0.136, -0.018)),  # I_C - i_P1   (rms 0.017)
}

# Tonry et al. 2012, Table 6: (x_SDSS - x_P1) as a function of (g - r)_P1
PS1_TO_SDSS = {
    'g': (0.013, 0.145, 0.019),
    'r': (-0.001, 0.004, 0.007),
    'i': (-0.005, 0.011, 0.010),
}

BAND_LABEL = {'B': 'B', 'V': 'V', 'R': 'R_C', 'I': 'I_C',
              'g': 'g', 'r': 'r', 'i': 'i'}


def reference_mags(cat, band, catalog, sdss=False):
    """
    Catalog magnitudes transformed into `band`.

    Returns (mags, color, system_label, transformation_label).
    `color` is catalog (g - r), used for the validity cut and color-trend
    diagnostic.
    """
    g, r, i = cat['g'], cat['r'], cat['i']
    color = g - r

    if catalog == 'PS1':
        if band in PS1_TO_JC:
            base, coeffs = PS1_TO_JC[band]
            mags = cat[base] + _quad(color, coeffs)
            return mags, color, BAND_LABEL[band], 'Tonry12'
        if band in ('g', 'r', 'i'):
            if sdss:
                return cat[band] + _quad(color, PS1_TO_SDSS[band]), color, \
                    band + '_SDSS', 'Tonry12'
            return cat[band].copy(), color, band + '_PS1', 'none'

    elif catalog == 'APASS':
        if band in ('B', 'V'):
            return cat[band].copy(), color, band, 'none'
        if band == 'R':
            return r - 0.2936 * (r - i) - 0.1439, color, 'R_C', 'Lupton05'
        if band == 'I':
            return r - 1.2444 * (r - i) - 0.3820, color, 'I_C', 'Lupton05'
        if band in ('g', 'r', 'i'):
            return cat[band].copy(), color, band + '_APASS', 'none'

    raise ValueError(f'No transformation for band {band!r} with catalog {catalog}')


# ---------------------------------------------------------------------------
# Filter-name handling
# ---------------------------------------------------------------------------

_CLEAR_NAMES = {'cv', 'clear', 'c', 'l', 'lum', 'luminance', 'open', 'none', 'unfiltered'}


def normalize_band(filt, cv_band='V'):
    """
    Map a FITS FILTER value to one of B, V, R, I, g, r, i.
    Uppercase single letters are Johnson-Cousins; lowercase / primed / 'p'
    suffixed are Sloan.  Unfiltered names map to `cv_band`.
    Returns (band, is_unfiltered).
    """
    if filt is None:
        raise ValueError('No FILTER keyword in header')
    f = str(filt).strip()
    fl = f.lower().replace("'", '').replace('_', '').replace(' ', '')

    if fl in _CLEAR_NAMES:
        return cv_band, True
    if f in ('B', 'V', 'R', 'I'):
        return f, False
    if fl in ('bj', 'johnsonb'):
        return 'B', False
    if fl in ('vj', 'johnsonv'):
        return 'V', False
    if fl in ('rc', 'cousinsr'):
        return 'R', False
    if fl in ('ic', 'cousinsi'):
        return 'I', False
    for s in ('g', 'r', 'i'):
        if fl in (s, s + 'p', 's' + s, 'sloan' + s):
            return s, False
    raise ValueError(f'Unrecognized FILTER value {filt!r}')


# ---------------------------------------------------------------------------
# Catalog queries
# ---------------------------------------------------------------------------

def field_cone(wcs, shape):
    """Center and radius of a cone that encloses the whole image."""
    ny, nx = shape
    center = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
    corners = wcs.pixel_to_world(np.array([0, nx, 0, nx]), np.array([0, 0, ny, ny]))
    radius = center.separation(corners).max()
    return center, radius


def _col(table, name):
    c = table[name]
    if hasattr(c, 'filled'):
        return np.asarray(c.filled(np.nan), dtype=float)
    return np.asarray(c, dtype=float)


def query_catalog(center, radius, catalog, sel_band, bright, faint,
                  faint_limit, min_stars):
    """
    Query PS1 or APASS around `center`.  The faint limit is relaxed one
    magnitude at a time (up to `faint_limit`) until at least `min_stars`
    stars are returned.  Returns a dict of float arrays
    (ra, dec, g, r, i, B, V) or None.
    """
    from astroquery.vizier import Vizier

    if catalog == 'PS1':
        vizcat = 'II/349/ps1'
        cols = ['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag']
        selcol = f'{sel_band}mag'
    else:
        vizcat = 'II/336/apass9'
        cols = ['RAJ2000', 'DEJ2000', 'Bmag', 'Vmag', "g'mag", "r'mag", "i'mag"]
        selcol = 'Vmag'

    table = None
    while True:
        v = Vizier(columns=cols, column_filters={selcol: f'{bright}..{faint}'},
                   row_limit=-1)
        try:
            res = v.query_region(center, radius=radius, catalog=vizcat)
        except Exception as e:
            print(f'  Catalog query failed: {e}')
            return None
        table = res[0] if len(res) else None
        n = 0 if table is None else len(table)
        if n >= min_stars or faint >= faint_limit:
            break
        faint += 1.0

    if table is None or len(table) == 0:
        return None

    out = {'ra': _col(table, 'RAJ2000'), 'dec': _col(table, 'DEJ2000')}
    if catalog == 'PS1':
        out['g'], out['r'], out['i'] = (_col(table, c) for c in ('gmag', 'rmag', 'imag'))
        out['B'] = out['V'] = np.full(len(table), np.nan)
    else:
        out['B'], out['V'] = _col(table, 'Bmag'), _col(table, 'Vmag')
        out['g'], out['r'], out['i'] = (_col(table, c) for c in ("g'mag", "r'mag", "i'mag"))
    return out


# ---------------------------------------------------------------------------
# Image measurements
# ---------------------------------------------------------------------------

def detect_sources(data, fwhm):
    """DAOStarFinder detections, thinned so no two are closer than 6 FWHM."""
    _, median, std = sigma_clipped_stats(data, sigma=5.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=15 * std, exclude_border=True)
    sources = daofind(data - median)
    if sources is None or len(sources) == 0:
        return np.empty((0, 2))
    pos = np.transpose((sources['xcentroid'], sources['ycentroid']))
    # Brightest first, so that in a close pair the brighter star is kept
    pos = pos[np.argsort(-np.asarray(sources['flux']))]
    keep = []
    min_d = 6 * fwhm
    for x, y in pos:
        if all((x - x0) ** 2 + (y - y0) ** 2 > min_d ** 2 for x0, y0 in keep):
            keep.append((x, y))
    return np.array(keep)


def measure_fwhm(data, positions, size=30):
    """Crude FWHM (pixels): median half-max extent of background-subtracted stars."""
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    bkg = Background2D(data, (100, 100), filter_size=(5, 5),
                       sigma_clip=SigmaClip(sigma=3.0),
                       bkg_estimator=MedianBackground())
    sub = data.astype(float) - bkg.background
    ny, nx = sub.shape
    vals = []
    for x, y in positions:
        x, y = int(round(x)), int(round(y))
        if x - size < 0 or x + size >= nx or y - size < 0 or y + size >= ny:
            continue
        cut = sub[y - size:y + size + 1, x - size:x + size + 1]
        yy, xx = np.where(cut >= 0.5 * cut.max())
        if len(xx):
            vals.append(((xx.max() - xx.min()) + (yy.max() - yy.min())) / 2.0)
    return float(np.median(vals)) if vals else np.nan


def aperture_mags(data, positions, fwhm, exptime):
    """Instrumental mags, -2.5 log10(e-/s).  Saturated or non-positive -> NaN."""
    radius = AP_RADIUS_FWHM * fwhm
    aps = CircularAperture(positions, r=radius)
    ann = CircularAnnulus(positions, r_in=radius + ANNULUS_GAP,
                          r_out=radius + ANNULUS_GAP + ANNULUS_WIDTH)
    phot = aperture_photometry(data, aps, method='exact')
    ap_sum = np.asarray(phot['aperture_sum'], dtype=float)

    ap_masks = aps.to_mask(method='center')
    ann_masks = ann.to_mask(method='center')
    bkg = np.empty(len(positions))
    for k in range(len(positions)):
        pix = ap_masks[k].get_values(data)
        if pix.size == 0 or np.any(pix > SATURATION):
            ap_sum[k] = np.nan
        sky = ann_masks[k].get_values(data)
        bkg[k] = sigma_clipped_stats(sky, sigma=3.0)[0] if sky.size else np.nan

    flux = GAIN * (ap_sum - bkg * aps.area)
    with np.errstate(invalid='ignore', divide='ignore'):
        mags = -2.5 * np.log10(flux / exptime)
    mags[~np.isfinite(mags)] = np.nan
    return mags


# ---------------------------------------------------------------------------
# Zero point
# ---------------------------------------------------------------------------

def compute_zeropoint(path, opts):
    data, header = fits.getdata(path, header=True)
    data = np.asarray(data, dtype=float)
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise ValueError('no valid WCS in header')

    band, unfiltered = normalize_band(header.get('FILTER'), opts.cvband)
    exptime = float(header['EXPTIME'])

    # FWHM: detect with header/default value, measure, re-detect
    fwhm0 = None
    for key in ('FWHM',):
        if key in header:
            try:
                fwhm0 = float(header[key])
            except (TypeError, ValueError):
                pass
    if fwhm0 is None and 'FWHMH' in header and 'FWHMV' in header:
        fwhm0 = 0.5 * (float(header['FWHMH']) + float(header['FWHMV']))
    if fwhm0 is None or not np.isfinite(fwhm0) or fwhm0 <= 0:
        fwhm0 = 8.0
    stars = detect_sources(data, fwhm0)
    fwhm = measure_fwhm(data, stars)
    if not np.isfinite(fwhm) or fwhm <= 1:
        fwhm = fwhm0
    stars = detect_sources(data, fwhm)
    if len(stars) < 3:
        raise ValueError(f'only {len(stars)} stars detected')

    # Catalog
    catalog = 'APASS' if opts.APASS else 'PS1'
    if catalog == 'PS1':
        sel = 'g' if band in ('B', 'g') else ('i' if band in ('I', 'i') else 'r')
        bright, faint = opts.bright, opts.faint
    else:
        sel = 'V'
        bright, faint = opts.bright_apass, opts.faint_apass
    center, radius = field_cone(wcs, data.shape)
    cat = query_catalog(center, radius, catalog, sel, bright, faint,
                        opts.faint_limit, opts.min_stars)
    if cat is None:
        raise ValueError(f'no {catalog} stars returned')

    ref, color, sys_label, trans = reference_mags(cat, band, catalog, opts.sdss)

    # Catalog stars -> pixels; keep those far enough from the edge
    cx, cy = wcs.all_world2pix(cat['ra'], cat['dec'], 0)
    margin = AP_RADIUS_FWHM * fwhm + ANNULUS_GAP + ANNULUS_WIDTH + 2
    ny, nx = data.shape
    ok = (np.isfinite(ref) & np.isfinite(color)
          & (color > opts.color_min) & (color < opts.color_max)
          & (cx > margin) & (cx < nx - margin) & (cy > margin) & (cy < ny - margin))
    cx, cy, ref, color = cx[ok], cy[ok], ref[ok], color[ok]
    if len(ref) < 3:
        raise ValueError('fewer than 3 usable catalog stars in the field')

    # Match to detections; drop blends (two catalog stars on one detection)
    tree = cKDTree(stars)
    dist, idx = tree.query(np.column_stack((cx, cy)))
    good = dist < 3 * fwhm
    counts = np.bincount(idx[good], minlength=len(stars))
    good &= counts[idx] == 1
    pos = stars[idx[good]]
    ref, color = ref[good], color[good]
    if len(ref) < 3:
        raise ValueError(f'only {len(ref)} catalog stars matched')

    inst = aperture_mags(data, pos, fwhm, exptime)
    offset = ref - inst
    fin = np.isfinite(offset)
    pos, ref, color, inst, offset = pos[fin], ref[fin], color[fin], inst[fin], offset[fin]

    clipped = sigma_clip(offset, sigma=opts.clip, maxiters=10,
                         cenfunc='median', stdfunc='mad_std')
    use = ~np.ma.getmaskarray(clipped)
    n = int(use.sum())
    if n < 3:
        raise ValueError(f'only {n} stars left after clipping')

    zp = float(np.mean(offset[use]))
    zp_std = float(np.std(offset[use], ddof=1))
    zp_sem = zp_std / np.sqrt(n)

    # Diagnostics only -- neither changes ZPMAG
    slope = float(np.polyfit(inst[use], ref[use], 1)[0])
    cterm = float(np.polyfit(color[use], offset[use], 1)[0])

    return dict(path=path, zp=zp, zp_std=zp_std, zp_sem=zp_sem, n=n,
                slope=slope, cterm=cterm, fwhm=fwhm, band=band,
                sys_label=sys_label, trans=trans, catalog=catalog,
                unfiltered=unfiltered, data=data,
                inst=inst[use], ref=ref[use], color=color[use],
                offset=offset[use], pos=pos[use])


def write_header(res):
    with fits.open(res['path'], mode='update') as hdul:
        h = hdul[0].header
        h['FWHM'] = (round(res['fwhm'], 2), '[pix] measured by ' + VERSION)
        h['ZPMAG'] = (round(res['zp'], 4), 'Zero point: m = -2.5log(e-/s) + ZPMAG')
        h['Z_ERR'] = (round(res['zp_std'], 4), 'Ensemble scatter (std) of ZP offsets')
        h['Z_SEM'] = (round(res['zp_sem'], 4), 'Standard error of mean ZP')
        h['Z_NSTAR'] = (res['n'], 'Comparison stars used in ZP')
        h['Z_CAT'] = (res['catalog'], 'Reference catalog')
        h['Z_BAND'] = (res['sys_label'], 'Band ZP is calibrated to')
        h['Z_TRANS'] = (res['trans'], 'Catalog->band transformation')
        h['Z_CTERM'] = (round(res['cterm'], 4), 'Diag: d(ZP)/d(g-r), not applied')
        h['Z_SLOPE'] = (round(res['slope'], 4), 'Diag: slope of cat vs inst mag')
        h['Z_CVER'] = (VERSION, 'Code that computed ZPMAG')
        hdul.flush()


def make_plots(res, opts):
    name = os.path.basename(res['path'])
    if opts.plot:
        plt.figure()
        plt.scatter(res['inst'], res['ref'], s=10)
        x = np.linspace(res['inst'].min(), res['inst'].max(), 2)
        plt.plot(x, x + res['zp'], 'r-', label=f"ZP = {res['zp']:.3f}")
        plt.xlabel('Instrumental magnitude')
        plt.ylabel(f"Catalog {res['sys_label']}")
        plt.title(name)
        plt.legend()
    if opts.plotZP:
        plt.figure()
        plt.scatter(res['ref'], res['offset'], s=10)
        plt.axhline(res['zp'], color='r')
        plt.xlabel(f"Catalog {res['sys_label']}")
        plt.ylabel('Catalog - instrumental')
        plt.title(name)
    if opts.plotColor:
        plt.figure()
        plt.scatter(res['color'], res['offset'], s=10)
        plt.axhline(res['zp'], color='r')
        xx = np.linspace(res['color'].min(), res['color'].max(), 2)
        b = res['zp'] - res['cterm'] * np.mean(res['color'])
        plt.plot(xx, b + res['cterm'] * xx, 'k--',
                 label=f"slope = {res['cterm']:.3f} mag/mag")
        plt.xlabel('Catalog (g - r)')
        plt.ylabel('Catalog - instrumental')
        plt.title(name)
        plt.legend()
    if opts.show:
        d = res['data']
        plt.figure()
        plt.imshow(d, cmap='Greys', origin='lower',
                   vmin=np.percentile(d, 5), vmax=np.percentile(d, 95))
        CircularAperture(res['pos'], r=AP_RADIUS_FWHM * res['fwhm']).plot(
            color='magenta', lw=0.8)
        plt.title(f'{name}: comparison stars used')
    if opts.plot or opts.plotZP or opts.plotColor or opts.show:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_files(path):
    directory = path if os.path.isdir(path) else (os.path.dirname(path) or '.')
    files = []
    for ext in ('fts', 'fit', 'fits'):
        files.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
    return sorted(set(files))


def main():
    p = argparse.ArgumentParser(
        description='Per-image ensemble zero points from PS1 or APASS.')
    p.add_argument('path', help='Directory of FITS images, or any file/glob in it')
    p.add_argument('--APASS', action='store_true',
                   help='Use APASS DR9 instead of Pan-STARRS1 (needed south of dec -30)')
    p.add_argument('--sdss', action='store_true',
                   help='For g/r/i frames with PS1, convert PS1 mags to SDSS (Tonry et al. 2012)')
    p.add_argument('--cvband', default='V', choices=['V', 'R', 'g', 'r'],
                   help='Band to calibrate unfiltered (CV/Clear) frames to (default V)')
    p.add_argument('--write', action='store_true', help='Write ZPMAG and diagnostics to the FITS header')
    p.add_argument('--rewrite', action='store_true',
                   help='Recompute even if the header already has a ZPMAG from this version')
    # Old flag names kept so existing scripts keep working
    p.add_argument('--writeAVG', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--rewriteAVG', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--PS1', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--plot', action='store_true', help='Plot catalog vs instrumental mags')
    p.add_argument('--plotZP', action='store_true', help='Plot ZP offset vs catalog mag')
    p.add_argument('--plotColor', action='store_true', help='Plot ZP offset vs catalog (g-r) color')
    p.add_argument('--show', action='store_true', help='Show image with comparison stars marked')
    p.add_argument('--bright', type=float, default=14.0, help='PS1 bright limit (PS1 saturates ~13-14)')
    p.add_argument('--faint', type=float, default=16.0, help='PS1 initial faint limit')
    p.add_argument('--bright_apass', type=float, default=10.0, help='APASS V bright limit')
    p.add_argument('--faint_apass', type=float, default=16.0, help='APASS V initial faint limit')
    p.add_argument('--faint_limit', type=float, default=19.0, help='Faintest the query will relax to')
    p.add_argument('--min_stars', type=int, default=30, help='Relax faint limit until this many catalog stars')
    p.add_argument('--color_min', type=float, default=-0.2, help='Min catalog (g-r) for comparison stars')
    p.add_argument('--color_max', type=float, default=1.4, help='Max catalog (g-r) for comparison stars')
    p.add_argument('--clip', type=float, default=2.5, help='Sigma for clipping ZP offsets')
    opts = p.parse_args()

    opts.write = opts.write or opts.writeAVG or opts.rewrite or opts.rewriteAVG
    opts.rewrite = opts.rewrite or opts.rewriteAVG

    files = find_files(opts.path)
    print(f'Found {len(files)} files')

    for f in files:
        name = os.path.basename(f)
        hdr = fits.getheader(f)
        if (not opts.rewrite and 'ZPMAG' in hdr
                and str(hdr.get('Z_CVER', '')).strip() == VERSION):
            print(f'{name}: ZPMAG already computed by {VERSION}, skipping (use --rewrite)')
            continue
        try:
            res = compute_zeropoint(f, opts)
        except Exception as e:
            print(f'{name}: skipped -- {e}')
            if opts.rewrite:
                # Don't leave a stale zero point behind
                with fits.open(f, mode='update') as hdul:
                    for k in ('ZPMAG', 'Z_ERR', 'Z_SEM', 'Z_NSTAR', 'Z_CAT', 'Z_BAND',
                              'Z_TRANS', 'Z_CTERM', 'Z_SLOPE', 'Z_CVER'):
                        hdul[0].header.pop(k, None)
                    hdul.flush()
            continue

        tag = ' (unfiltered)' if res['unfiltered'] else ''
        print(f"{name}: ZP = {res['zp']:.3f} +/- {res['zp_sem']:.3f} "
              f"(scatter {res['zp_std']:.3f}, N = {res['n']}) "
              f"-> {res['sys_label']}{tag} via {res['catalog']}/{res['trans']}; "
              f"color trend {res['cterm']:+.3f} mag/mag, slope {res['slope']:.3f}")
        if opts.write:
            write_header(res)
        make_plots(res, opts)


if __name__ == '__main__':
    main()
