from astropy.io import fits
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def find_exp(file_dir):
    patterns = ['*.fts', '*.fts.fz', '*.fit', '*.fits']
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(file_dir, p))

    if not files:
        raise FileNotFoundError(f"No FITS files found in {file_dir} to determine exposure time.")

    for f in files:
        try:
            with fits.open(f, memmap=True, mode='readonly',
                           do_not_scale_image_data=True, lazy_load_hdus=True) as hdul:
                expt = hdul[0].header.get("EXPTIME")
                if expt is None and len(hdul) > 1:
                    expt = hdul[1].header.get("EXPTIME")
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}. Skipping.")
            continue

        if expt is not None:
            return expt

    raise ValueError(f"No EXPTIME keyword found in any FITS file in {file_dir}.")

def find_filts_fast(file_dir):
    patterns = ['*.fts', '*.fts.fz', '*.fit', '*.fits']
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(file_dir, p))

    if not files:
        raise FileNotFoundError(f"No FITS files found in {file_dir} to determine filters.")

    filters = set()
    for f in files:
        try:
            with fits.open(f, memmap=True, mode='readonly',
                           do_not_scale_image_data=True, lazy_load_hdus=True) as hdul:
                filt = hdul[0].header.get("FILTER")
                if not filt and len(hdul) > 1:
                    filt = hdul[1].header.get("FILTER")
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}. Skipping.")
            continue

        if filt:
            filters.add(filt)

    return sorted(filters)

def find_images(file_dir, filt):
        path = file_dir + '/'
        patterns = ['*.fts', '*.fts.fz', '*.fit', '*.fits']
        files = []
        for p in patterns:
            files += glob.glob(os.path.join(path, '*_' + filt + '_') + p)
        return sorted(files)